// minfer M1 — CPU fp32 reference forward pass for Qwen2.5 (correctness oracle).
// Slow on purpose: weights stay quantized in the mmap and are dequantized per
// matvec row. Works in token-ID space so no tokenizer is needed here.
//
//   cpu_infer <model.gguf> <n_gen> <id0> <id1> ...
// Prints the greedy continuation token IDs (space separated) on stdout.
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace minfer;

struct Config {
    int n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_vocab;
    float rope_base, rms_eps;
};

// y[out] = W_row_j . x   for a weight tensor W with ne0=in, ne1=out.
static void matvec(const Tensor& W, const float* x, float* y) {
    const int64_t in = W.ne[0], out = W.ne[1];
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < out; j++) {
        float row[16384];                 // >= max in-dim (n_ff=8960)
        GGUF::dequant_row(W, j, row);
        float acc = 0.f;
        for (int64_t k = 0; k < in; k++) acc += row[k] * x[k];
        y[j] = acc;
    }
}

static void add_bias(float* y, const Tensor& b, int n) {
    static thread_local std::vector<float> buf;
    buf.resize(n); GGUF::dequant_row(b, 0, buf.data());
    for (int i = 0; i < n; i++) y[i] += buf[i];
}

static void rmsnorm(const float* x, const Tensor& w, int n, float eps, float* out) {
    static thread_local std::vector<float> wb;
    wb.resize(n); GGUF::dequant_row(w, 0, wb.data());
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * wb[i];
}

// NEOX-style RoPE over a vector of n_head*head_dim, pairing (i, i+hd/2).
static void rope(float* v, int n_head, int hd, int pos, float base) {
    int half = hd / 2;
    for (int h = 0; h < n_head; h++) {
        float* p = v + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            float theta = pos * std::pow(base, -2.f * i / hd);
            float c = std::cos(theta), s = std::sin(theta);
            float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * c - x1 * s;
            p[i + half] = x0 * s + x1 * c;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gen> <id...>\n", argv[0]); return 1; }
    GGUF g(argv[1]);
    int n_gen = atoi(argv[2]);
    std::vector<int> ids;
    for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));

    Config c;
    c.n_layer   = (int)g.meta_num("qwen2.block_count", 0);
    c.n_embd    = (int)g.meta_num("qwen2.embedding_length", 0);
    c.n_ff      = (int)g.meta_num("qwen2.feed_forward_length", 0);
    c.n_head    = (int)g.meta_num("qwen2.attention.head_count", 0);
    c.n_head_kv = (int)g.meta_num("qwen2.attention.head_count_kv", 0);
    c.head_dim  = c.n_embd / c.n_head;
    c.rope_base = (float)g.meta_num("qwen2.rope.freq_base", 1e6);
    c.rms_eps   = (float)g.meta_num("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    c.n_vocab   = (int)g.tokens().size();
    const int kv_dim = c.n_head_kv * c.head_dim;
    const int gqa = c.n_head / c.n_head_kv;

    // gather tensors
    const Tensor& tok_emb = g.get("token_embd.weight");
    const Tensor& out_w   = g.get("output.weight");
    const Tensor& out_norm= g.get("output_norm.weight");
    struct Layer { const Tensor *an,*wq,*bq,*wk,*bk,*wv,*bv,*wo,*fn,*wg,*wu,*wd; };
    std::vector<Layer> L(c.n_layer);
    for (int i = 0; i < c.n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        L[i] = { &g.get(p+"attn_norm.weight"),
                 &g.get(p+"attn_q.weight"), &g.get(p+"attn_q.bias"),
                 &g.get(p+"attn_k.weight"), &g.get(p+"attn_k.bias"),
                 &g.get(p+"attn_v.weight"), &g.get(p+"attn_v.bias"),
                 &g.get(p+"attn_output.weight"),
                 &g.get(p+"ffn_norm.weight"),
                 &g.get(p+"ffn_gate.weight"), &g.get(p+"ffn_up.weight"), &g.get(p+"ffn_down.weight") };
    }

    const int max_seq = (int)ids.size() + n_gen;
    // KV cache: [layer][pos * kv_dim]
    std::vector<std::vector<float>> kc(c.n_layer), vc(c.n_layer);
    for (int i = 0; i < c.n_layer; i++) { kc[i].resize((size_t)max_seq*kv_dim); vc[i].resize((size_t)max_seq*kv_dim); }

    std::vector<float> h(c.n_embd), xb(c.n_embd), q(c.n_head*c.head_dim), k(kv_dim), v(kv_dim),
                       att(c.n_head*c.head_dim), proj(c.n_embd), gbuf(c.n_ff), ubuf(c.n_ff),
                       logits(c.n_vocab);

    auto forward = [&](int token, int pos) -> int {
        GGUF::dequant_row(tok_emb, token, h.data());           // embedding
        for (int l = 0; l < c.n_layer; l++) {
            Layer& ly = L[l];
            rmsnorm(h.data(), *ly.an, c.n_embd, c.rms_eps, xb.data());
            matvec(*ly.wq, xb.data(), q.data()); add_bias(q.data(), *ly.bq, c.n_head*c.head_dim);
            matvec(*ly.wk, xb.data(), k.data()); add_bias(k.data(), *ly.bk, kv_dim);
            matvec(*ly.wv, xb.data(), v.data()); add_bias(v.data(), *ly.bv, kv_dim);
            rope(q.data(), c.n_head, c.head_dim, pos, c.rope_base);
            rope(k.data(), c.n_head_kv, c.head_dim, pos, c.rope_base);
            memcpy(&kc[l][(size_t)pos*kv_dim], k.data(), kv_dim*4);
            memcpy(&vc[l][(size_t)pos*kv_dim], v.data(), kv_dim*4);
            float inv = 1.f / std::sqrt((float)c.head_dim);
            for (int hh = 0; hh < c.n_head; hh++) {
                int kvh = hh / gqa;
                const float* qh = &q[hh*c.head_dim];
                std::vector<float> sc(pos+1);
                float mx = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const float* kt = &kc[l][(size_t)t*kv_dim + kvh*c.head_dim];
                    float d = 0; for (int i = 0; i < c.head_dim; i++) d += qh[i]*kt[i];
                    sc[t] = d*inv; if (sc[t] > mx) mx = sc[t];
                }
                float sum = 0; for (int t = 0; t <= pos; t++) { sc[t] = std::exp(sc[t]-mx); sum += sc[t]; }
                float* oh = &att[hh*c.head_dim];
                for (int i = 0; i < c.head_dim; i++) oh[i] = 0;
                for (int t = 0; t <= pos; t++) {
                    const float* vt = &vc[l][(size_t)t*kv_dim + kvh*c.head_dim];
                    float p = sc[t]/sum;
                    for (int i = 0; i < c.head_dim; i++) oh[i] += p*vt[i];
                }
            }
            matvec(*ly.wo, att.data(), proj.data());
            for (int i = 0; i < c.n_embd; i++) h[i] += proj[i];   // residual
            // FFN (SwiGLU)
            rmsnorm(h.data(), *ly.fn, c.n_embd, c.rms_eps, xb.data());
            matvec(*ly.wg, xb.data(), gbuf.data());
            matvec(*ly.wu, xb.data(), ubuf.data());
            for (int i = 0; i < c.n_ff; i++) { float x = gbuf[i]; gbuf[i] = (x/(1.f+std::exp(-x)))*ubuf[i]; }
            matvec(*ly.wd, gbuf.data(), proj.data());
            for (int i = 0; i < c.n_embd; i++) h[i] += proj[i];
        }
        rmsnorm(h.data(), out_norm, c.n_embd, c.rms_eps, xb.data());
        matvec(out_w, xb.data(), logits.data());
        int best = 0; float bv2 = logits[0];
        for (int i = 1; i < c.n_vocab; i++) if (logits[i] > bv2) { bv2 = logits[i]; best = i; }
        return best;
    };

    fprintf(stderr, "minfer-cpu: qwen2 L=%d d=%d ff=%d H=%d/%d hd=%d vocab=%d\n",
            c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab);

    int pos = 0, next = -1;
    for (size_t i = 0; i < ids.size(); i++) next = forward(ids[i], pos++);  // prefill
    for (int t = 0; t < n_gen; t++) {
        printf("%d ", next);
        next = forward(next, pos++);
    }
    printf("\n");
    return 0;
}
