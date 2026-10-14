// bench/breakdown.cu — Phase 0: where does cuda_infer's decode step spend its time?
// (nsys/CUPTI needs root on this Jetson.) Reuses the engine's real kernels verbatim by
// including src/cuda_infer.cu, builds Qwen2.5-1.5B-shaped synthetic q4_0 weights (timing
// doesn't depend on values), and times three CUDA graphs of one decode step at a fixed
// position: FULL (exactly record()'s sequence, monolithic attention, fp32 KV), GEMV-only,
// and OTHER-only. Then each GEMV shape on its own, cycling through 28 distinct matrices
// so small ones can't sit in the 2 MB L2.
//
//   nvcc -O3 -arch=sm_87 -o build/breakdown bench/breakdown.cu && ./build/breakdown [pos]
#define main minfer_main_unused
#include "../src/cuda_infer.cu"
#undef main

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (uint8_t)h;
    }
}
__global__ void fill_half(__half* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half(v);
}
__global__ void fill_f32(float* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
}
static DTensor mk(int in, int out, uint32_t seed) {
    size_t nblk = (size_t)out * (in / 32);
    uint8_t* q; __half* s; CK(cudaMalloc(&q, nblk * 16)); CK(cudaMalloc(&s, nblk * 2));
    fill_bytes<<<256, 256>>>(q, nblk * 16, seed); fill_half<<<256, 256>>>(s, nblk, 0.002f);
    return DTensor{GT_Q4_0, in, out, q, s};
}
static float* mkf(size_t n, float v) { float* p; CK(cudaMalloc(&p, n * 4)); fill_f32<<<64, 256>>>(p, n, v); return p; }

int main(int argc, char** argv) {
    const int NL = 28, E = 1536, FF = 8960, NH = 12, NKV = 2, HD = 128, KVD = NKV * HD, V = 151936, GQA = NH / NKV;
    const int pos = argc > 1 ? atoi(argv[1]) : 32, maxseq = 4096;
    const float eps = 1e-6f, base = 1e6f, inv = 1.f / sqrtf((float)HD);
    struct Ly { DTensor q, k, v, o, g, u, d; float *an, *fn, *bq, *bk, *bv; };
    std::vector<Ly> ly(NL);
    for (int l = 0; l < NL; l++) {
        uint32_t s = 1000 * l;
        ly[l] = {mk(E, E, s + 1), mk(E, KVD, s + 2), mk(E, KVD, s + 3), mk(E, E, s + 4),
                 mk(E, FF, s + 5), mk(E, FF, s + 6), mk(FF, E, s + 7),
                 mkf(E, 1.f), mkf(E, 1.f), mkf(E, 0.f), mkf(KVD, 0.f), mkf(KVD, 0.f)};
    }
    DTensor lm = mk(E, V, 7);
    float *h = mkf(E, 0.1f), *xb = mkf(E, 0.f), *q = mkf(E, 0.f), *att = mkf(E, 0.f), *proj = mkf(E, 0.f);
    float *gb = mkf(FF, 0.f), *ub = mkf(FF, 0.f), *logits = mkf(V, 0.f), *kb = mkf(KVD, 0.f), *vb = mkf(KVD, 0.f);
    float *kc = mkf((size_t)NL * maxseq * KVD, 0.01f), *vc = mkf((size_t)NL * maxseq * KVD, 0.01f);
    float *on = mkf(E, 1.f), *hfix = mkf(E, 0.1f);
    int8_t* xq; __half* xd; CK(cudaMalloc(&xq, FF)); CK(cudaMalloc(&xd, FF / 32 * 2));
    int *dpos, *dtok; CK(cudaMalloc(&dpos, 4)); CK(cudaMalloc(&dtok, 4));
    CK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    // one decode step = record() with ATTN=0, KVQ=0, LMHEAD=4 (minus record_token/advance).
    // what: 1 = GEMVs only, 2 = everything else, 3 = both.  only: restrict GEMVs to one shape.
    auto step = [&](cudaStream_t s, int what, char only) {
        bool G = what & 1, O = what & 2;
        auto D = [&](char tag, float* y, DTensor W, const float* b) {
            if (G && (!only || only == tag)) gemv_dp4a<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, xq, xd, b);
        };
        if (O) { CK(cudaMemcpyAsync(h, hfix, E * 4, cudaMemcpyDeviceToDevice, s)); }  // keep h finite
        for (int l = 0; l < NL; l++) {
            Ly& y = ly[l];
            float* kc_l = kc + (size_t)l * maxseq * KVD; float* vc_l = vc + (size_t)l * maxseq * KVD;
            if (O) { rmsnorm_k<<<1, 256, 0, s>>>(xb, h, y.an, E, eps); quant_x_q8<<<E / 32, 32, 0, s>>>(xb, xq, xd, E); }
            D('q', q, y.q, y.bq); D('k', kb, y.k, y.bk); D('v', vb, y.v, y.bv);
            if (O) {
                rope_k<<<(NH * HD / 2 + 127) / 128, 128, 0, s>>>(q, NH, HD, dpos, base);
                rope_k<<<(KVD / 2 + 127) / 128, 128, 0, s>>>(kb, NKV, HD, dpos, base);
                kv_store<<<(KVD + 127) / 128, 128, 0, s>>>(kc_l, vc_l, kb, vb, dpos, KVD);
                attn_head<<<NH, HD, (HD + maxseq) * 4, s>>>(att, q, kc_l, vc_l, dpos, NH, HD, KVD, GQA, inv);
                quant_x_q8<<<E / 32, 32, 0, s>>>(att, xq, xd, E);
            }
            D('o', proj, y.o, nullptr);
            if (O) {
                residual_k<<<(E + 127) / 128, 128, 0, s>>>(h, proj, E);
                rmsnorm_k<<<1, 256, 0, s>>>(xb, h, y.fn, E, eps); quant_x_q8<<<E / 32, 32, 0, s>>>(xb, xq, xd, E);
            }
            D('g', gb, y.g, nullptr); D('u', ub, y.u, nullptr);
            if (O) { swiglu_k<<<(FF + 127) / 128, 128, 0, s>>>(gb, ub, FF); quant_x_q8<<<FF / 32, 32, 0, s>>>(gb, xq, xd, FF); }
            D('d', proj, y.d, nullptr);
            if (O) residual_k<<<(E + 127) / 128, 128, 0, s>>>(h, proj, E);
        }
        if (O) { rmsnorm_k<<<1, 256, 0, s>>>(xb, h, on, E, eps); quant_x_q8<<<E / 32, 32, 0, s>>>(xb, xq, xd, E); }
        D('l', logits, lm, nullptr);
        if (O) argmax_k<<<1, 256, 0, s>>>(dtok, logits, V);
    };
    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto graph_ms = [&](int what, char only) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); step(s, what, only);
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        size_t nn; CK(cudaGraphGetNodes(g, nullptr, &nn));
        for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int N = 30; cudaEventRecord(a, s);
        for (int i = 0; i < N; i++) CK(cudaGraphLaunch(ge, s));
        cudaEventRecord(b, s); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        return std::make_pair(ms / N, (int)nn);
    };
    for (float t = 0; t < 1500.f; ) t += graph_ms(3, 0).first * 33;   // ramp DVFS clocks (unpinned) first
    const double MB = 1e6, Bq = (double)E * E * 18 / 32, Bkv = (double)E * KVD * 18 / 32, Bff = (double)E * FF * 18 / 32,
                 Blm = (double)E * V * 18 / 32, Btot = NL * (2 * Bq + 2 * Bkv + 3 * Bff) + Blm;
    printf("synthetic Qwen2.5-1.5B q4_0 decode step at pos=%d  (weights %.1f MB/token)\n\n", pos, Btot / MB);
    auto full = graph_ms(3, 0), gemv = graph_ms(1, 0), other = graph_ms(2, 0);
    printf("  FULL step   %6.3f ms  (%4d nodes)  -> %5.1f tok/s, %5.1f GB/s effective\n",
           full.first, full.second, 1e3 / full.first, Btot / full.first / 1e6);
    printf("  GEMV only   %6.3f ms  (%4d nodes)  -> %5.1f GB/s while in GEMVs\n", gemv.first, gemv.second, Btot / gemv.first / 1e6);
    printf("  OTHER only  %6.3f ms  (%4d nodes)  -> %.2f us/node\n", other.first, other.second, other.first * 1e3 / other.second);
    printf("  GEMV+OTHER  %6.3f ms  vs FULL %.3f ms\n", gemv.first + other.first, full.first);
    printf("  roofline    %6.3f ms  (at 100 GB/s) -> %5.1f tok/s\n\n", Btot / 100e9 * 1e3, 100e9 / Btot);
    printf("  %-8s %8s %9s %7s %10s\n", "matrix", "MB", "us/call", "GB/s", "ms/token");
    struct { char t; const char* n; double b; int c; } sh[] = {
        {'q', "Wq", Bq, NL}, {'k', "Wk", Bkv, NL}, {'v', "Wv", Bkv, NL}, {'o', "Wo", Bq, NL},
        {'g', "Wgate", Bff, NL}, {'u', "Wup", Bff, NL}, {'d', "Wdown", Bff, NL}, {'l', "lm_head", Blm, 1}};
    for (auto& x : sh) {
        auto r = graph_ms(1, x.t);
        double us = r.first * 1e3 / x.c;
        printf("  %-8s %8.2f %9.1f %7.1f %10.3f\n", x.n, x.b / MB, us, x.b / us / 1e3, r.first);
    }
    return 0;
}
