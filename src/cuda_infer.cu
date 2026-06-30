// minfer M2 — naive CUDA port of the Qwen2.5 forward pass.
// Correctness, not speed. Weights stay quantized resident on device (~1 GB,
// same footprint as llama.cpp's q4_0) and are dequantized inside the kernels;
// activations are fp32. Gate: greedy token IDs match the CPU reference / oracle.
//
//   cuda_infer <model.gguf> <n_gen> <id0> <id1> ...
#include "gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>

using namespace minfer;

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

// device-resident tensor. For repacked Q4_0, `data` is contiguous 16-byte quant
// blocks (16-aligned) and `scales` holds the per-block fp16 d; otherwise raw bytes.
struct DTensor { int type, in, out; const uint8_t* data; const __half* scales; };

// ---------------- kernels ----------------

// dequantize one element e (0..255) of a Q6_K superblock at p
__device__ __forceinline__ float deq_q6k(const uint8_t* p, int e) {
    const int8_t* sc = (const int8_t*)(p + 192);
    float d = __half2float(*(const __half*)(p + 208));
    int adv = (e & 128) ? 1 : 0;                 // second 128-chunk advances pointers
    const uint8_t* QL = p + (adv ? 64 : 0);
    const uint8_t* QH = p + 128 + (adv ? 32 : 0);
    const int8_t*  SC = sc + (adv ? 8 : 0);
    int o = e & 127, l = o & 31, grp = o >> 5, is = l / 16, qv, sci;
    if      (grp == 0) { qv = ((QL[l]      & 0xF) | (((QH[l] >> 0) & 3) << 4)) - 32; sci = is + 0; }
    else if (grp == 1) { qv = ((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32; sci = is + 2; }
    else if (grp == 2) { qv = ((QL[l]      >>  4) | (((QH[l] >> 4) & 3) << 4)) - 32; sci = is + 4; }
    else               { qv = ((QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4)) - 32; sci = is + 6; }
    return d * SC[sci] * qv;
}

// y[j] = dequant(W_row_j) . x  (+ bias[j]).  One WARP per output row, lanes
// stride the K dimension, warp-shuffle reduction. blockDim must be a multiple
// of 32; WPB = blockDim/32 rows are computed per block.
__global__ void gemv(float* y, DTensor W, const float* x, const float* bias) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int j = blockIdx.x * wpb + warp;
    if (j >= W.out) return;
    float part = 0.f;
    if (W.type == GT_Q4_0) {
        // lane = element-within-block (0..31). All 32 lanes cooperate on one
        // block per iteration, so x[b*32+lane] is 32 consecutive floats
        // (coalesced) and the 16 quant bytes are a coalesced broadcast read.
        int nb = W.in / 32;
        size_t rowbase = (size_t)j * nb;
        const uint8_t* qbytes = W.data;            // 16 bytes per block, contiguous
        for (int b = 0; b < nb; b++) {
            float d = __half2float(W.scales[rowbase + b]);
            uint8_t byte = qbytes[(rowbase + b) * 16 + (lane & 15)];
            int q = (lane < 16) ? (byte & 0xF) : (byte >> 4);
            part += (q - 8) * d * x[b * 32 + lane];
        }
    } else {  // GT_Q6_K
        int nsb = W.in / 256;
        const uint8_t* base = W.data + (size_t)j * nsb * 210;
        for (int sb = 0; sb < nsb; sb++) {
            const uint8_t* p = base + (size_t)sb * 210; const float* xb = x + sb * 256;
            for (int e = lane; e < 256; e += 32) part += deq_q6k(p, e) * xb[e];
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
    if (lane == 0) y[j] = part + (bias ? bias[j] : 0.f);
}

// symmetric per-32-block int8 quantization of the activation vector (Q8-style):
// xq[i] = round(x[i] * 127/amax_block), xd[g] = amax_block/127.  <<<n/32, 32>>>
__global__ void quant_x_q8(const float* x, int8_t* xq, __half* xd, int n) {
    int g = blockIdx.x, i = threadIdx.x, idx = g * 32 + i;
    float v = (idx < n) ? x[idx] : 0.f, a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffffu, a, o));
    a = __shfl_sync(0xffffffffu, a, 0);            // amax broadcast to all lanes
    float inv = (a > 0.f) ? 127.f / a : 0.f;
    if (idx < n) xq[idx] = (int8_t)__float2int_rn(v * inv);
    if (i == 0) xd[g] = __float2half(a / 127.f);
}

// dp4a Q4_0 GEMV: dot product runs in INT8 hardware MACs (4/instruction), one
// float multiply per 32-block. Ported from llm-optimization/gemv_q4_dp4a.
// W repacked (W.data = 16-byte blocks, W.scales = fp16). x pre-quantized to Xq/Xd.
__global__ void gemv_dp4a(float* y, DTensor W, const int8_t* Xq, const __half* Xd,
                          const float* bias) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int NB = W.in / 32; size_t rowbase = (size_t)row * NB;
    const uint8_t* Wq = W.data;
    float sum = 0.f;
    for (int b = lane; b < NB; b += 32) {
        size_t blk = rowbase + b;
        float dw = __half2float(W.scales[blk]), dx = __half2float(Xd[b]);
        uint4 p = *reinterpret_cast<const uint4*>(Wq + blk * 16);   // 16 quant bytes
        // load the 32 int8 activations as 2 wide uint4 (2 loads) not 8 int loads:
        // the kernel is L1-instruction-throughput bound, so fewer/wider loads win.
        uint4 xa = *reinterpret_cast<const uint4*>(Xq + b * 32);
        uint4 xc = *reinterpret_cast<const uint4*>(Xq + b * 32 + 16);
        const int xq[8] = {(int)xa.x,(int)xa.y,(int)xa.z,(int)xa.w,(int)xc.x,(int)xc.y,(int)xc.z,(int)xc.w};
        const unsigned M = 0x0F0F0F0Fu;
        // SIMD nibble unpack: mask whole words (no per-byte extraction). Q4_0 order:
        // low nibbles of bytes 0..15 = elems 0..15, high nibbles = elems 16..31.
        // (word & M) holds 4 nibbles (0..15) -> dp4a vs the matching 4 x. The -8 is
        // folded out: sum (n-8)*x = sum n*x - 8*sum x.
        int acc = 0, sx = 0;
        acc = __dp4a((int)(p.x & M), xq[0], acc); acc = __dp4a((int)(p.y & M), xq[1], acc);
        acc = __dp4a((int)(p.z & M), xq[2], acc); acc = __dp4a((int)(p.w & M), xq[3], acc);
        acc = __dp4a((int)((p.x >> 4) & M), xq[4], acc); acc = __dp4a((int)((p.y >> 4) & M), xq[5], acc);
        acc = __dp4a((int)((p.z >> 4) & M), xq[6], acc); acc = __dp4a((int)((p.w >> 4) & M), xq[7], acc);
        #pragma unroll
        for (int k = 0; k < 8; k++) sx = __dp4a(0x01010101, xq[k], sx);   // sum of the 32 x
        sum += dw * dx * (float)(acc - 8 * sx);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) y[row] = sum + (bias ? bias[row] : 0.f);
}

// dp4a Q8_0 GEMV. Linear layout (32 int8 + fp16 d per block) => fully coalesced:
// lane b reads 32 contiguous weight bytes, warp covers 1 KB/iter. Used for the
// lm_head after requantizing Q6_K->Q8_0 at load (Q8_0 is finer than Q6_K, so the
// weight values are near-lossless; the only approx is the x->q8 activation).
__global__ void gemv_q8_dp4a(float* y, DTensor W, const int8_t* Xq, const __half* Xd,
                             const float* bias) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int NB = W.in / 32; size_t rowbase = (size_t)row * NB;
    const int8_t* Wq = (const int8_t*)W.data;
    float sum = 0.f;
    for (int b = lane; b < NB; b += 32) {
        size_t blk = rowbase + b;
        float dw = __half2float(W.scales[blk]), dx = __half2float(Xd[b]);
        uint4 wa = *reinterpret_cast<const uint4*>(Wq + blk * 32);          // wide loads:
        uint4 wb = *reinterpret_cast<const uint4*>(Wq + blk * 32 + 16);     // 2 uint4 weight
        uint4 xa = *reinterpret_cast<const uint4*>(Xq + b * 32);            // + 2 uint4 x
        uint4 xc = *reinterpret_cast<const uint4*>(Xq + b * 32 + 16);
        const int w32[8] = {(int)wa.x,(int)wa.y,(int)wa.z,(int)wa.w,(int)wb.x,(int)wb.y,(int)wb.z,(int)wb.w};
        const int x32[8] = {(int)xa.x,(int)xa.y,(int)xa.z,(int)xa.w,(int)xc.x,(int)xc.y,(int)xc.z,(int)xc.w};
        int acc = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) acc = __dp4a(w32[k], x32[k], acc);
        sum += dw * dx * (float)acc;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) y[row] = sum + (bias ? bias[row] : 0.f);
}

// ===================== BATCHED (multi-token) kernels =====================
// Process T tokens at once. The point on a memory-bound box: a weight matrix is
// streamed ONCE and reused for all T activation vectors (unpack the weight block
// once, then T dp4a). This is the "verify K tokens for the price of 1" primitive
// that makes speculative decoding worthwhile. Layout: activations/hidden are
// row-major [T][dim]; logits [T][vocab].
#define SPEC_MAX 8                       // max tokens per batched pass (draft window+1)

// quant T activation vectors -> Xq[T][dim] int8, Xd[T][dim/32] fp16.  grid(dim/32, T)
__global__ void quant_x_q8_batch(const float* X, int8_t* Xq, __half* Xd, int dim, int T) {
    int t = blockIdx.y, g = blockIdx.x, i = threadIdx.x, idx = g * 32 + i;
    const float* x = X + (size_t)t * dim;
    float v = (idx < dim) ? x[idx] : 0.f, a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffffu, a, o));
    a = __shfl_sync(0xffffffffu, a, 0);
    float inv = (a > 0.f) ? 127.f / a : 0.f;
    if (idx < dim) Xq[(size_t)t * dim + idx] = (int8_t)__float2int_rn(v * inv);
    if (i == 0)    Xd[(size_t)t * (dim / 32) + g] = __float2half(a / 127.f);
}

// Q4_0 dp4a, T tokens (T is a compile-time template param so the accumulator is
// exactly T registers -> higher occupancy, lower per-token slope). Weight block
// unpacked once into wp[8], reused for all T.
template<int T>
__global__ void gemv_dp4a_batch(float* Y, DTensor W, const int8_t* Xq, const __half* Xd,
                                const float* bias, int in) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int NB = in / 32; size_t rowbase = (size_t)row * NB;
    const uint8_t* Wq = W.data;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; t++) sum[t] = 0.f;
    for (int b = lane; b < NB; b += 32) {
        size_t blk = rowbase + b;
        float dw = __half2float(W.scales[blk]);
        uint4 p = *reinterpret_cast<const uint4*>(Wq + blk * 16);
        const unsigned M = 0x0F0F0F0Fu;
        int nw[8];                                       // 8 nibble-words (0..15), unpacked once
        nw[0]=(int)(p.x&M); nw[1]=(int)(p.y&M); nw[2]=(int)(p.z&M); nw[3]=(int)(p.w&M);
        nw[4]=(int)((p.x>>4)&M); nw[5]=(int)((p.y>>4)&M); nw[6]=(int)((p.z>>4)&M); nw[7]=(int)((p.w>>4)&M);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float dx = __half2float(Xd[(size_t)t * NB + b]);
            uint4 xa = *reinterpret_cast<const uint4*>(Xq + (size_t)t * in + b * 32);
            uint4 xc = *reinterpret_cast<const uint4*>(Xq + (size_t)t * in + b * 32 + 16);
            const int xq[8] = {(int)xa.x,(int)xa.y,(int)xa.z,(int)xa.w,(int)xc.x,(int)xc.y,(int)xc.z,(int)xc.w};
            int acc = 0, sx = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) acc = __dp4a(nw[k], xq[k], acc);
            #pragma unroll
            for (int k = 0; k < 8; k++) sx  = __dp4a(0x01010101, xq[k], sx);
            sum[t] += dw * dx * (float)(acc - 8 * sx);   // -8 folded: sum(n-8)x = sum n x - 8 sum x
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) Y[(size_t)t * W.out + row] = s + (bias ? bias[row] : 0.f);
    }
}

// Q8_0 dp4a, T tokens (lm_head verify), templated on T. Block (32 int8) loaded once.
template<int T>
__global__ void gemv_q8_dp4a_batch(float* Y, DTensor W, const int8_t* Xq, const __half* Xd,
                                   const float* bias, int in) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int NB = in / 32; size_t rowbase = (size_t)row * NB;
    const int8_t* Wq = (const int8_t*)W.data;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; t++) sum[t] = 0.f;
    for (int b = lane; b < NB; b += 32) {
        size_t blk = rowbase + b;
        float dw = __half2float(W.scales[blk]);
        uint4 wa = *reinterpret_cast<const uint4*>(Wq + blk * 32);          // weight once, wide
        uint4 wb = *reinterpret_cast<const uint4*>(Wq + blk * 32 + 16);
        const int w32[8] = {(int)wa.x,(int)wa.y,(int)wa.z,(int)wa.w,(int)wb.x,(int)wb.y,(int)wb.z,(int)wb.w};
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float dx = __half2float(Xd[(size_t)t * NB + b]);
            uint4 xa = *reinterpret_cast<const uint4*>(Xq + (size_t)t * in + b * 32);
            uint4 xc = *reinterpret_cast<const uint4*>(Xq + (size_t)t * in + b * 32 + 16);
            const int x32[8] = {(int)xa.x,(int)xa.y,(int)xa.z,(int)xa.w,(int)xc.x,(int)xc.y,(int)xc.z,(int)xc.w};
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) acc = __dp4a(w32[k], x32[k], acc);
            sum[t] += dw * dx * (float)acc;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) Y[(size_t)t * W.out + row] = s + (bias ? bias[row] : 0.f);
    }
}

// host dispatch: pick the compile-time T instantiation at launch.
static void launch_dp4a_batch(float* Y, DTensor W, const int8_t* Xq, const __half* Xd,
                              const float* bias, int T, int in, cudaStream_t s) {
    int grid = (W.out + 3) / 4;
    #define CASE(N) case N: gemv_dp4a_batch<N><<<grid, 128, 0, s>>>(Y, W, Xq, Xd, bias, in); break;
    switch (T) { CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
    #undef CASE
}
static void launch_q8_dp4a_batch(float* Y, DTensor W, const int8_t* Xq, const __half* Xd,
                                 const float* bias, int T, int in, cudaStream_t s) {
    int grid = (W.out + 3) / 4;
    #define CASE(N) case N: gemv_q8_dp4a_batch<N><<<grid, 128, 0, s>>>(Y, W, Xq, Xd, bias, in); break;
    switch (T) { CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
    #undef CASE
}

// embedding gather for T tokens -> H[T][n_embd].  grid(1, T)
__global__ void embed_q4_batch(float* H, DTensor W, const int* toks, int T) {
    int t = blockIdx.y, tid = threadIdx.x, nt = blockDim.x, nb = W.in / 32, token = toks[t];
    const uint8_t* base = W.data + (size_t)token * nb * 18;
    float* h = H + (size_t)t * W.in;
    for (int b = tid; b < nb; b += nt) {
        const uint8_t* p = base + (size_t)b * 18;
        float d = __half2float(*(const __half*)p);
        const uint8_t* qs = p + 2; float* y = h + b * 32;
        for (int k = 0; k < 16; k++) { y[k] = ((qs[k] & 0xF) - 8) * d; y[k + 16] = ((qs[k] >> 4) - 8) * d; }
    }
}

// RMSNorm each of T rows independently.  grid(T), block 256
__global__ void rmsnorm_batch(float* out, const float* x, const float* w, int n, float eps, int T) {
    int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* xr = x + (size_t)t * n; float* outr = out + (size_t)t * n;
    __shared__ float red[256]; float s = 0;
    for (int i = tid; i < n; i += nt) s += xr[i] * xr[i];
    red[tid] = s; __syncthreads();
    for (int k = nt / 2; k > 0; k >>= 1) { if (tid < k) red[tid] += red[tid + k]; __syncthreads(); }
    float scale = rsqrtf(red[0] / n + eps);
    for (int i = tid; i < n; i += nt) outr[i] = xr[i] * scale * w[i];
}

// NEOX RoPE for T tokens at positions pos0..pos0+T-1.  grid(ceil(nh*hd/2/128), T)
__global__ void rope_batch(float* V, int n_head, int hd, int pos0, float base, int rowstride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, half = hd / 2, total = n_head * half;
    if (idx >= total) return;
    int t = blockIdx.y, h = idx / half, i = idx % half, pos = pos0 + t;
    float* p = V + (size_t)t * rowstride + (size_t)h * hd;
    float theta = pos * powf(base, -2.f * i / hd), c = cosf(theta), s = sinf(theta);
    float x0 = p[i], x1 = p[i + half];
    p[i] = x0 * c - x1 * s; p[i + half] = x0 * s + x1 * c;
}

// store T tokens' k,v into the cache at slots pos0..pos0+T-1.  grid(ceil(kv_dim/128), T)
__global__ void kv_store_batch(float* kc_l, float* vc_l, const float* kbuf, const float* vbuf,
                               int pos0, int kv_dim, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= kv_dim) return;
    int t = blockIdx.y; size_t off = (size_t)(pos0 + t) * kv_dim + i;
    kc_l[off] = kbuf[(size_t)t * kv_dim + i]; vc_l[off] = vbuf[(size_t)t * kv_dim + i];
}

// causal GQA attention for T query tokens. grid(n_head, T), block hd. Query t (at
// absolute position pos0+t) attends cache positions 0..pos0+t (inclusive).
__global__ void attn_batch(float* out, const float* q, const float* kc, const float* vc,
                           int pos0, int n_head, int hd, int kv_dim, int gqa, float inv, int T) {
    int h = blockIdx.x, t = blockIdx.y; if (h >= n_head) return;
    int tid = threadIdx.x, nt = blockDim.x, pos = pos0 + t, kvh = h / gqa;
    extern __shared__ float sm[];
    float* qsh = sm; float* sc = sm + hd; __shared__ float red[128];
    const float* qh = q + (size_t)t * n_head * hd + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) qsh[i] = qh[i];
    __syncthreads();
    for (int u = tid; u <= pos; u += nt) {
        const float* kt = kc + (size_t)u * kv_dim + kvh * hd;
        float d = 0;
        #pragma unroll 4
        for (int i = 0; i < hd; i++) d += qsh[i] * kt[i];
        sc[u] = d * inv;
    }
    __syncthreads();
    float m = -1e30f;
    for (int u = tid; u <= pos; u += nt) m = fmaxf(m, sc[u]);
    red[tid] = m; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sum = 0;
    for (int u = tid; u <= pos; u += nt) { float e = expf(sc[u] - mx); sc[u] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float denom = red[0]; __syncthreads();
    if (tid < hd) {
        float a = 0;
        for (int u = 0; u <= pos; u++) a += sc[u] * vc[(size_t)u * kv_dim + kvh * hd + tid];
        out[(size_t)t * n_head * hd + (size_t)h * hd + tid] = a / denom;
    }
}

// Reconstruct the 16 6-bit weight values (0..63) of 16-group `ig` (natural
// elements 16*ig .. 16*ig+15) of a Q6_K superblock at p, into q16[0..15].
// Same integer math as deq_q6k, just hoisted: a 16-group shares grp/is/adv, so
// all the branch selection happens once instead of per element.
__device__ __forceinline__ void q6k_qvals(const uint8_t* p, int ig, int q16[16]) {
    int adv = ig >> 3;                       // ig>=8 -> second 128-chunk
    const uint8_t* QL = p + (adv ? 64 : 0);
    const uint8_t* QH = p + 128 + (adv ? 32 : 0);
    int grp = (ig >> 1) & 3, l0 = (ig & 1) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int l = l0 + k;
        switch (grp) {
            case 0:  q16[k] = (QL[l]      & 0xF) | (((QH[l] >> 0) & 3) << 4); break;
            case 1:  q16[k] = (QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4); break;
            case 2:  q16[k] = (QL[l]      >>  4) | (((QH[l] >> 4) & 3) << 4); break;
            default: q16[k] = (QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4); break;
        }
    }
}

// DEBUG: Q6_K GEMV using the hoisted q6k_qvals dequant but fp32 activations
// (no int8). Isolates the dequant logic from activation quantization error.
__global__ void gemv_q6k_fp32(float* y, DTensor W, const float* x, const float* bias) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int nsb = W.in / 256;
    const uint8_t* base = W.data + (size_t)row * nsb * 210;
    float part = 0.f;
    for (int gidx = lane; gidx < nsb * 16; gidx += 32) {
        int sb = gidx >> 4, ig = gidx & 15;
        const uint8_t* p = base + (size_t)sb * 210;
        float d  = __half2float(*(const __half*)(p + 208));
        int   sc = ((const int8_t*)(p + 192))[ig];
        int q16[16]; q6k_qvals(p, ig, q16);
        const float* xb = x + sb * 256 + ig * 16;
        float g = 0.f;
        #pragma unroll
        for (int k = 0; k < 16; k++) g += (q16[k] - 32) * xb[k];
        part += d * sc * g;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
    if (lane == 0) y[row] = part + (bias ? bias[row] : 0.f);
}

// dp4a Q6_K GEMV (lm_head). One WARP per output row; the 32 lanes split the
// nsb*16 sixteen-groups of the row. Per group: rebuild 16 int8 weights, run the
// dot in INT8 hardware MACs against the q8 activation, fold the Q6_K -32 offset
// via sum(xq). w[e]=d*sc*(q-32), x[e]=dx*xq[e]  =>  group += d*sc*dx*(dot-32*sum).
// Exact Q6_K weights (no requant); only approximation is x->q8 (same as elsewhere).
__global__ void gemv_q6k_dp4a(float* y, DTensor W, const int8_t* Xq, const __half* Xd,
                              const float* bias) {
    int wpb = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * wpb + warp; if (row >= W.out) return;
    int nsb = W.in / 256;
    const uint8_t* base = W.data + (size_t)row * nsb * 210;
    float part = 0.f;
    for (int gidx = lane; gidx < nsb * 16; gidx += 32) {
        int sb = gidx >> 4, ig = gidx & 15;
        const uint8_t* p = base + (size_t)sb * 210;
        float d  = __half2float(*(const __half*)(p + 208));
        int   sc = ((const int8_t*)(p + 192))[ig];
        float dx = __half2float(Xd[sb * 8 + (ig >> 1)]);
        int q16[16]; q6k_qvals(p, ig, q16);
        const int8_t* xq = Xq + sb * 256 + ig * 16;   // 16 activations (aligned)
        int dot = 0, ssum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int wpack = (q16[4*k+0] & 0xFF) | ((q16[4*k+1] & 0xFF) << 8) |
                        ((q16[4*k+2] & 0xFF) << 16) | ((q16[4*k+3] & 0xFF) << 24);
            int xpack = *reinterpret_cast<const int*>(xq + 4*k);
            dot  = __dp4a(wpack, xpack, dot);
            ssum = __dp4a(0x01010101, xpack, ssum);    // sum of the 4 xq (for -32 fold)
        }
        part += d * sc * dx * (float)(dot - 32 * ssum);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
    if (lane == 0) y[row] = part + (bias ? bias[row] : 0.f);
}

// embedding gather: dequant Q4_0 row[*dtok] -> h   (token read from device)
__global__ void embed_q4(float* h, DTensor W, const int* dtok) {
    int tid = threadIdx.x, nt = blockDim.x, nb = W.in / 32, token = *dtok;
    const uint8_t* base = W.data + (size_t)token * nb * 18;
    for (int b = tid; b < nb; b += nt) {
        const uint8_t* p = base + (size_t)b * 18;
        float d = __half2float(*(const __half*)p);
        const uint8_t* qs = p + 2; float* y = h + b * 32;
        for (int k = 0; k < 16; k++) { y[k] = ((qs[k] & 0xF) - 8) * d; y[k + 16] = ((qs[k] >> 4) - 8) * d; }
    }
}

__global__ void rmsnorm_k(float* out, const float* x, const float* w, int n, float eps) {
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float red[256]; float s = 0;
    for (int i = tid; i < n; i += nt) s += x[i] * x[i];
    red[tid] = s; __syncthreads();
    for (int k = nt / 2; k > 0; k >>= 1) { if (tid < k) red[tid] += red[tid + k]; __syncthreads(); }
    float scale = rsqrtf(red[0] / n + eps);
    for (int i = tid; i < n; i += nt) out[i] = x[i] * scale * w[i];
}

// NEOX RoPE over n_head*hd, pairing (i, i+hd/2)   (position read from device)
__global__ void rope_k(float* v, int n_head, int hd, const int* dpos, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, half = hd / 2, total = n_head * half;
    if (idx >= total) return;
    int h = idx / half, i = idx % half, pos = *dpos; float* p = v + (size_t)h * hd;
    float theta = pos * powf(base, -2.f * i / hd), c = cosf(theta), s = sinf(theta);
    float x0 = p[i], x1 = p[i + half];
    p[i] = x0 * c - x1 * s; p[i + half] = x0 * s + x1 * c;
}

// store this step's k,v into the KV cache at slot *dpos
__global__ void kv_store(float* kc_l, float* vc_l, const float* kbuf, const float* vbuf,
                         const int* dpos, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= kv_dim) return;
    size_t off = (size_t)(*dpos) * kv_dim + i;
    kc_l[off] = kbuf[i]; vc_l[off] = vbuf[i];
}

// parallel causal GQA attention: one BLOCK per head, blockDim==hd threads cooperate.
// shared = q_head[hd] + scores[pos+1]. Phase1 thread-per-position dot, phase2 block
// softmax, phase3 thread-per-output-dim weighted V sum.
__global__ void attn_head(float* out, const float* q, const float* kc, const float* vc,
                          const int* dpos, int n_head, int hd, int kv_dim, int gqa, float inv) {
    int h = blockIdx.x; if (h >= n_head) return;
    int tid = threadIdx.x, nt = blockDim.x, pos = *dpos, kvh = h / gqa;
    extern __shared__ float sm[];
    float* qsh = sm;          // hd
    float* sc  = sm + hd;     // pos+1
    __shared__ float red[128];

    const float* qh = q + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) qsh[i] = qh[i];
    __syncthreads();

    for (int t = tid; t <= pos; t += nt) {            // phase 1: scores
        const float* kt = kc + (size_t)t * kv_dim + kvh * hd;
        float d = 0;
        #pragma unroll 4
        for (int i = 0; i < hd; i++) d += qsh[i] * kt[i];
        sc[t] = d * inv;
    }
    __syncthreads();

    float m = -1e30f;                                  // phase 2: softmax
    for (int t = tid; t <= pos; t += nt) m = fmaxf(m, sc[t]);
    red[tid] = m; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sum = 0;
    for (int t = tid; t <= pos; t += nt) { float e = expf(sc[t] - mx); sc[t] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float denom = red[0]; __syncthreads();

    if (tid < hd) {                                    // phase 3: weighted V
        float a = 0;
        for (int t = 0; t <= pos; t++) a += sc[t] * vc[(size_t)t * kv_dim + kvh * hd + tid];
        out[(size_t)h * hd + tid] = a / denom;
    }
}

// ---------------- int8 KV cache (KVQ) ----------------
// Halve→quarter the bytes attention re-reads each step at depth: store K and V as
// int8 with one fp16 scale per (token, kv_head). Quantization is per-head symmetric
// (amax/127). Near-lossless; the whole win is fewer DRAM bytes in the attention loop.
// block = hd (128) threads; one block per kv_head reduces that head's amax.
__device__ __forceinline__ float blk_amax(float v, float* red, int tid, int hd) {
    red[tid] = fabsf(v); __syncthreads();
    for (int s = hd / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float a = red[0]; __syncthreads();
    return a;
}
// single-token store (decode graph): pos from device. grid(n_kv), block hd.
__global__ void kv_store_q8(int8_t* kcq_l, __half* kcs_l, int8_t* vcq_l, __half* vcs_l,
                            const float* kbuf, const float* vbuf, const int* dpos,
                            int kv_dim, int n_kv, int hd) {
    int kvh = blockIdx.x, tid = threadIdx.x, pos = *dpos, idx = kvh * hd + tid;
    __shared__ float red[128];
    float kv = kbuf[idx], vv = vbuf[idx];
    float ks = blk_amax(kv, red, tid, hd) / 127.f, vs = blk_amax(vv, red, tid, hd) / 127.f;
    float ki = ks > 0.f ? 1.f / ks : 0.f, vi = vs > 0.f ? 1.f / vs : 0.f;
    size_t off = (size_t)pos * kv_dim + idx;
    kcq_l[off] = (int8_t)__float2int_rn(kv * ki);
    vcq_l[off] = (int8_t)__float2int_rn(vv * vi);
    if (tid == 0) { kcs_l[(size_t)pos * n_kv + kvh] = __float2half(ks); vcs_l[(size_t)pos * n_kv + kvh] = __float2half(vs); }
}
// T-token store (batched verify): pos0..pos0+T-1. grid(n_kv, T), block hd.
__global__ void kv_store_batch_q8(int8_t* kcq_l, __half* kcs_l, int8_t* vcq_l, __half* vcs_l,
                                  const float* kbuf, const float* vbuf, int pos0,
                                  int kv_dim, int n_kv, int hd, int T) {
    int kvh = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, pos = pos0 + t, idx = kvh * hd + tid;
    __shared__ float red[128];
    float kv = kbuf[(size_t)t * kv_dim + idx], vv = vbuf[(size_t)t * kv_dim + idx];
    float ks = blk_amax(kv, red, tid, hd) / 127.f, vs = blk_amax(vv, red, tid, hd) / 127.f;
    float ki = ks > 0.f ? 1.f / ks : 0.f, vi = vs > 0.f ? 1.f / vs : 0.f;
    size_t off = (size_t)pos * kv_dim + idx;
    kcq_l[off] = (int8_t)__float2int_rn(kv * ki);
    vcq_l[off] = (int8_t)__float2int_rn(vv * vi);
    if (tid == 0) { kcs_l[(size_t)pos * n_kv + kvh] = __float2half(ks); vcs_l[(size_t)pos * n_kv + kvh] = __float2half(vs); }
}
// attention over an int8 cache. Dequant K/V on load (scale per token,kv_head). The
// dot/weighted-sum stay fp32; only the bytes read shrink 4×.  decode: grid(n_head).
__global__ void attn_head_q8(float* out, const float* q, const int8_t* kcq, const __half* kcs,
                             const int8_t* vcq, const __half* vcs, const int* dpos,
                             int n_head, int hd, int kv_dim, int n_kv, int gqa, float inv) {
    int h = blockIdx.x; if (h >= n_head) return;
    int tid = threadIdx.x, nt = blockDim.x, pos = *dpos, kvh = h / gqa;
    extern __shared__ float sm[];
    float* qsh = sm; float* sc = sm + hd; __shared__ float red[128];
    const float* qh = q + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) qsh[i] = qh[i];
    __syncthreads();
    for (int t = tid; t <= pos; t += nt) {
        const int8_t* kt = kcq + (size_t)t * kv_dim + kvh * hd;
        float sk = __half2float(kcs[(size_t)t * n_kv + kvh]);
        float d = 0;
        #pragma unroll 4
        for (int i = 0; i < hd; i++) d += qsh[i] * (float)kt[i];
        sc[t] = d * sk * inv;
    }
    __syncthreads();
    float m = -1e30f;
    for (int t = tid; t <= pos; t += nt) m = fmaxf(m, sc[t]);
    red[tid] = m; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sum = 0;
    for (int t = tid; t <= pos; t += nt) { float e = expf(sc[t] - mx); sc[t] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float denom = red[0]; __syncthreads();
    if (tid < hd) {
        float a = 0;
        for (int t = 0; t <= pos; t++) {
            float sv = __half2float(vcs[(size_t)t * n_kv + kvh]);
            a += sc[t] * (float)vcq[(size_t)t * kv_dim + kvh * hd + tid] * sv;
        }
        out[(size_t)h * hd + tid] = a / denom;
    }
}
// batched (T query tokens) attention over the int8 cache. grid(n_head, T).
__global__ void attn_batch_q8(float* out, const float* q, const int8_t* kcq, const __half* kcs,
                              const int8_t* vcq, const __half* vcs, int pos0,
                              int n_head, int hd, int kv_dim, int n_kv, int gqa, float inv, int T) {
    int h = blockIdx.x, t = blockIdx.y; if (h >= n_head) return;
    int tid = threadIdx.x, nt = blockDim.x, pos = pos0 + t, kvh = h / gqa;
    extern __shared__ float sm[];
    float* qsh = sm; float* sc = sm + hd; __shared__ float red[128];
    const float* qh = q + (size_t)t * n_head * hd + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) qsh[i] = qh[i];
    __syncthreads();
    for (int u = tid; u <= pos; u += nt) {
        const int8_t* kt = kcq + (size_t)u * kv_dim + kvh * hd;
        float sk = __half2float(kcs[(size_t)u * n_kv + kvh]);
        float d = 0;
        #pragma unroll 4
        for (int i = 0; i < hd; i++) d += qsh[i] * (float)kt[i];
        sc[u] = d * sk * inv;
    }
    __syncthreads();
    float m = -1e30f;
    for (int u = tid; u <= pos; u += nt) m = fmaxf(m, sc[u]);
    red[tid] = m; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sum = 0;
    for (int u = tid; u <= pos; u += nt) { float e = expf(sc[u] - mx); sc[u] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float denom = red[0]; __syncthreads();
    if (tid < hd) {
        float a = 0;
        for (int u = 0; u <= pos; u++) {
            float sv = __half2float(vcs[(size_t)u * n_kv + kvh]);
            a += sc[u] * (float)vcq[(size_t)u * kv_dim + kvh * hd + tid] * sv;
        }
        out[(size_t)t * n_head * hd + (size_t)h * hd + tid] = a / denom;
    }
}

// ---------------- M12: split-KV (flash-style) attention ----------------
// The single-token attn_head launches only n_head (=12) blocks -> ~12% occupancy on
// 8 SMs, so its O(pos) serial loops balloon at depth (23->129us, the long-ctx slide).
// Fix: split the KV range into CHUNK-sized pieces, one block per (head,chunk) -> ~5x
// more blocks fills the SMs. Each block computes a partial softmax (local max m, local
// denom l, un-normalized weighted-V o relative to m); a combine kernel merges chunks
// with online-softmax rescaling (numerically exact vs the monolithic softmax, up to
// fp reassociation). grid(n_head, max_chunks); blocks past the current pos early-out.
__global__ void attn_partial(const float* q, const float* kc, const float* vc, const int* dpos,
                             float* m_part, float* l_part, float* o_part,
                             int n_head, int hd, int kv_dim, int gqa, float inv, int CHUNK, int max_chunks) {
    int h = blockIdx.x, c = blockIdx.y, pos = *dpos;
    if (c * CHUNK > pos) return;                          // chunk entirely beyond current pos
    int tid = threadIdx.x, nt = blockDim.x, kvh = h / gqa;
    int c0 = c * CHUNK, c1 = min(c0 + CHUNK, pos + 1), len = c1 - c0;
    extern __shared__ float sm[];
    float* qsh = sm; float* sc = sm + hd; __shared__ float red[128];
    const float* qh = q + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) qsh[i] = qh[i];
    __syncthreads();
    for (int j = tid; j < len; j += nt) {                 // scores within the chunk
        const float* kt = kc + (size_t)(c0 + j) * kv_dim + kvh * hd;
        float d = 0;
        #pragma unroll 4
        for (int i = 0; i < hd; i++) d += qsh[i] * kt[i];
        sc[j] = d * inv;
    }
    __syncthreads();
    float m = -1e30f;                                     // local max
    for (int j = tid; j < len; j += nt) m = fmaxf(m, sc[j]);
    red[tid] = m; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float ml = red[0]; __syncthreads();
    float ls = 0;                                         // exp + local denom
    for (int j = tid; j < len; j += nt) { float e = expf(sc[j] - ml); sc[j] = e; ls += e; }
    red[tid] = ls; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float ll = red[0]; __syncthreads();
    if (tid < hd) {                                       // un-normalized partial output
        float a = 0;
        for (int j = 0; j < len; j++) a += sc[j] * vc[(size_t)(c0 + j) * kv_dim + kvh * hd + tid];
        o_part[((size_t)h * max_chunks + c) * hd + tid] = a;
    }
    if (tid == 0) { m_part[h * max_chunks + c] = ml; l_part[h * max_chunks + c] = ll; }
}
// merge the per-chunk partials for each head. grid(n_head), block hd.
__global__ void attn_combine(float* out, const float* m_part, const float* l_part, const float* o_part,
                             const int* dpos, int n_head, int hd, int CHUNK, int max_chunks) {
    int h = blockIdx.x, tid = threadIdx.x, pos = *dpos;
    int nchunks = (pos + 1 + CHUNK - 1) / CHUNK;
    float M = -1e30f;
    for (int c = 0; c < nchunks; c++) M = fmaxf(M, m_part[h * max_chunks + c]);
    float denom = 0;
    for (int c = 0; c < nchunks; c++) denom += l_part[h * max_chunks + c] * expf(m_part[h * max_chunks + c] - M);
    if (tid < hd) {
        float a = 0;
        for (int c = 0; c < nchunks; c++)
            a += o_part[((size_t)h * max_chunks + c) * hd + tid] * expf(m_part[h * max_chunks + c] - M);
        out[(size_t)h * hd + tid] = a / denom;
    }
}

// M13: record the step's input token into out[*step] and bump the counter, all on
// device. Placed at the head of the decode graph so a run of K self-advancing graph
// launches can proceed with NO host sync between them; the host copies all K tokens
// once per chunk instead of doing a blocking D2H every token (which stalled the GPU
// between launches — decode was partly host-issue-bound).
__global__ void record_token(int* out, int* step, const int* tok) {
    out[*step] = *tok; (*step)++;
}

__global__ void residual_k(float* h, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) h[i] += x[i];
}
__global__ void swiglu_k(float* g, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = g[i]; g[i] = (x / (1.f + expf(-x))) * u[i]; }
}

// greedy argmax over logits -> *out (device-side, so it can feed the next step)
__global__ void argmax_k(int* out, const float* logits, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float bv[256]; __shared__ int bi[256];
    float best = -1e30f; int bidx = 0;
    for (int i = tid; i < n; i += nt) { float v = logits[i]; if (v > best) { best = v; bidx = i; } }
    bv[tid] = best; bi[tid] = bidx; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s && bv[tid + s] > bv[tid]) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[0] = bi[0];
}
__global__ void advance_k(int* dpos) { if (threadIdx.x == 0) (*dpos)++; }

// ---------------- host driver ----------------

// Evict a tensor's mmap pages from page cache after it's been copied to device.
// On Jetson's unified memory the GGUF mmap (file-backed, read-only) and the device
// copies share one pool; without this the page cache grows to the full model size
// in parallel with device allocs and a big model (7B) OOMs mid-load. The mapping is
// PROT_READ MAP_PRIVATE so pages are never dirty — MADV_DONTNEED just drops them.
static void release_pages(const Tensor& t) {
    if (!t.data) return;
    long ps = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)t.data, start = a & ~(uintptr_t)(ps - 1);
    madvise((void*)start, (a - start) + t.nbytes(), MADV_DONTNEED);
}

static const uint8_t* upload(const Tensor& t) {
    uint8_t* d; CK(cudaMalloc(&d, t.nbytes()));
    CK(cudaMemcpy(d, t.data, t.nbytes(), cudaMemcpyHostToDevice));
    release_pages(t);
    return d;
}
static DTensor dt(const Tensor& t) {
    return DTensor{(int)t.type, (int)t.ne[0], (int)t.rows(), upload(t), nullptr};
}
// repack a Q4_0 weight into contiguous {fp16 scales, 16-byte quant blocks} for
// coalesced vectorized loads in the GEMV.
static DTensor dt_q4r(const Tensor& t) {
    int in = (int)t.ne[0], out = (int)t.rows(), nb = in / 32;
    size_t nblk = (size_t)out * nb;
    std::vector<uint8_t> quants(nblk * 16);
    std::vector<uint16_t> scales(nblk);          // raw fp16 bits
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t* blk = t.data + b * 18;
        memcpy(&scales[b], blk, 2);
        memcpy(&quants[b * 16], blk + 2, 16);
    }
    uint8_t* dq; CK(cudaMalloc(&dq, quants.size()));
    CK(cudaMemcpy(dq, quants.data(), quants.size(), cudaMemcpyHostToDevice));
    __half* ds; CK(cudaMalloc(&ds, nblk * 2));
    CK(cudaMemcpy(ds, scales.data(), nblk * 2, cudaMemcpyHostToDevice));
    release_pages(t);
    return DTensor{GT_Q4_0, in, out, dq, ds};
}
// load a 2D weight matrix: Q4_0 stays repacked Q4_0 (native dp4a kernel); any other
// quant (Q4_1 ffn_down in some GGUFs, Q5_*, Q6_K) is requantized to Q8_0 at load so
// the existing dp4a GEMV consumes it. Costs a little extra VRAM (Q8_0 > Q4_x).
static DTensor dt_w(const Tensor& t);   // fwd decl; defined after dt_q8_0r
// requantize any tensor (here the Q6_K lm_head) to repacked Q8_0: dequant each
// row to fp32, then per-32-block symmetric int8. {int8 quants, fp16 scales}.
static DTensor dt_q8_0r(const Tensor& t) {
    int in = (int)t.ne[0], out = (int)t.rows(), nb = in / 32;
    size_t nblk = (size_t)out * nb;
    std::vector<int8_t> quants(nblk * 32);
    std::vector<__half> scales(nblk);
    std::vector<float> row(in);
    for (int r = 0; r < out; r++) {
        GGUF::dequant_row(t, r, row.data());
        for (int b = 0; b < nb; b++) {
            float amax = 0.f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(row[b * 32 + j]));
            float d = amax / 127.f, inv = (d > 0.f) ? 1.f / d : 0.f;
            size_t blk = (size_t)r * nb + b;
            scales[blk] = __float2half(d);
            for (int j = 0; j < 32; j++)
                quants[blk * 32 + j] = (int8_t)lrintf(row[b * 32 + j] * inv);
        }
    }
    int8_t* dq; CK(cudaMalloc(&dq, quants.size()));
    CK(cudaMemcpy(dq, quants.data(), quants.size(), cudaMemcpyHostToDevice));
    __half* ds; CK(cudaMalloc(&ds, nblk * 2));
    CK(cudaMemcpy(ds, scales.data(), nblk * 2, cudaMemcpyHostToDevice));
    release_pages(t);
    return DTensor{GT_Q8_0, in, out, (const uint8_t*)dq, ds};
}
// requantize any tensor to repacked Q4_0 ({fp16 scales, 16-byte quant blocks}),
// the format gemv_dp4a consumes natively. Lets us coerce Q4_1 (the ffn_down quant
// in llama.cpp's q4_0 recipe) into the q4 path with NO VRAM bloat (Q4_0 18 B/blk <
// Q4_1 20 B/blk), unlike going to Q8_0. Drops Q4_1's per-block min (symmetric q4),
// a small accuracy cost on ffn_down — fine when memory is the binding constraint.
static DTensor dt_q4_0r_from(const Tensor& t) {
    int in = (int)t.ne[0], out = (int)t.rows(), nb = in / 32;
    size_t nblk = (size_t)out * nb;
    std::vector<uint8_t> quants(nblk * 16);
    std::vector<uint16_t> scales(nblk);
    std::vector<float> row(in);
    for (int r = 0; r < out; r++) {
        GGUF::dequant_row(t, r, row.data());
        for (int b = 0; b < nb; b++) {
            const float* y = row.data() + b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(y[j]));
            float d = amax / 7.f, inv = (d > 0.f) ? 1.f / d : 0.f;
            size_t blk = (size_t)r * nb + b;
            __half hd = __float2half(d); memcpy(&scales[blk], &hd, 2);
            for (int j = 0; j < 16; j++) {
                int q0 = (int)lrintf(y[j]      * inv) + 8; q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
                int q1 = (int)lrintf(y[j + 16] * inv) + 8; q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
                quants[blk * 16 + j] = (uint8_t)(q0 | (q1 << 4));
            }
        }
    }
    uint8_t* dq; CK(cudaMalloc(&dq, quants.size()));
    CK(cudaMemcpy(dq, quants.data(), quants.size(), cudaMemcpyHostToDevice));
    __half* ds; CK(cudaMalloc(&ds, nblk * 2));
    CK(cudaMemcpy(ds, scales.data(), nblk * 2, cudaMemcpyHostToDevice));
    release_pages(t);
    return DTensor{GT_Q4_0, in, out, dq, ds};
}
static DTensor dt_w(const Tensor& t) { return (t.type == GT_Q4_0) ? dt_q4r(t) : dt_q4_0r_from(t); }
// ---- GPT2 byte-level BPE detokenizer ----
// Qwen2 vocab pieces are stored as UTF-8 of a reversible byte->unicode map (the
// GPT2 "bytes_to_unicode" trick: control/space bytes are remapped to printable
// codepoints so the BPE alphabet has no whitespace). To get real text back we
// reverse it: decode each piece's UTF-8 to codepoints, map each codepoint to its
// original byte. cp2byte must be sized >= 324 (max remapped codepoint = 323).
static void build_byte_decoder(int* cp2byte) {
    for (int i = 0; i < 324; i++) cp2byte[i] = -1;
    std::vector<int> bs, cs;
    for (int b = '!';  b <= '~';  b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
    }
    for (size_t i = 0; i < bs.size(); i++) cp2byte[cs[i]] = bs[i];
}

// Decode one vocab piece (UTF-8 of remapped codepoints) into raw bytes, appended to out.
static void detok_piece(const std::string& s, const int* cp2byte, std::string& out) {
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = (unsigned char)s[i];
        int cp, len;
        if      (c < 0x80)        { cp = c;        len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; len = 4; }
        else { i++; continue; }
        for (int k = 1; k < len && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < 324 && cp2byte[cp] >= 0) out.push_back((char)cp2byte[cp]);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gen> <id...>\n", argv[0]); return 1; }
    GGUF g(argv[1]);
    int n_gen = atoi(argv[2]);
    std::vector<int> ids; for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));

    int n_layer = (int)g.meta_num("qwen2.block_count", 0);
    int n_embd  = (int)g.meta_num("qwen2.embedding_length", 0);
    int n_ff    = (int)g.meta_num("qwen2.feed_forward_length", 0);
    int n_head  = (int)g.meta_num("qwen2.attention.head_count", 0);
    int n_kv    = (int)g.meta_num("qwen2.attention.head_count_kv", 0);
    int hd      = n_embd / n_head, kv_dim = n_kv * hd, gqa = n_head / n_kv;
    float rope_base = (float)g.meta_num("qwen2.rope.freq_base", 1e6);
    float eps = (float)g.meta_num("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    int n_vocab = (int)g.tokens().size();
    int maxseq = (int)ids.size() + n_gen + SPEC_MAX;   // +SPEC_MAX: verify writes ahead

    // Output: detokenized text by default (streamed, flushed per token); RAWIDS=1
    // prints token IDs instead (the format the correctness gate parses).
    bool raw_ids = getenv("RAWIDS") && atoi(getenv("RAWIDS"));
    int eos_id = (int)g.meta_num("tokenizer.ggml.eos_token_id", 151645);  // stop in text mode
    const std::vector<std::string>& vocab = g.tokens();
    int cp2byte[324]; build_byte_decoder(cp2byte);
    auto emit = [&](int id) {
        if (raw_ids) { printf("%d ", id); return; }
        if (id >= 0 && id < (int)vocab.size()) {
            std::string out; detok_piece(vocab[id], cp2byte, out);
            fputs(out.c_str(), stdout); fflush(stdout);
        }
    };

    // lm_head Q6_K variant: 0=scalar(fp32,exact) 1=fp32-fast 2=Q6_K-dp4a 3=Q8_0-dp4a(fast).
    // Default 3: int8 lm_head (like llama.cpp) -> ~45 tok/s. Set LMHEAD=0 for the
    // byte-exact fp32 reference (~26 tok/s).
    int lmhead = getenv("LMHEAD") ? atoi(getenv("LMHEAD")) : 4;  // 4=q4 lm_head: ½ bytes, +4–5%
    DTensor tok_emb = dt(g.get("token_embd.weight"));
    // lm_head: keep raw Q6_K, except mode 3 requantizes to Q8_0 and mode 4 to q4_0.
    // The lm_head is ~25% of the bytes read per token; Q8_0 (248MB) was chosen for a
    // coalesced layout, but q4_0 (131MB, mode 4) reads ~half the bytes — on a
    // memory-bound decode that should be FASTER, at some logit-precision cost.
    DTensor out_w = (lmhead == 3) ? dt_q8_0r(g.get("output.weight"))
                  : (lmhead == 4) ? dt_q4_0r_from(g.get("output.weight"))
                                  : dt(g.get("output.weight"));
    const float* out_norm = (const float*)upload(g.get("output_norm.weight"));
    struct L { DTensor wq, wk, wv, wo, wg, wu, wd; const float *an, *bq, *bk, *bv, *fn; };
    std::vector<L> ly(n_layer);
    for (int i = 0; i < n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        ly[i].an = (const float*)upload(g.get(p + "attn_norm.weight"));
        ly[i].wq = dt_w(g.get(p + "attn_q.weight")); ly[i].bq = (const float*)upload(g.get(p + "attn_q.bias"));
        ly[i].wk = dt_w(g.get(p + "attn_k.weight")); ly[i].bk = (const float*)upload(g.get(p + "attn_k.bias"));
        ly[i].wv = dt_w(g.get(p + "attn_v.weight")); ly[i].bv = (const float*)upload(g.get(p + "attn_v.bias"));
        ly[i].wo = dt_w(g.get(p + "attn_output.weight"));
        ly[i].fn = (const float*)upload(g.get(p + "ffn_norm.weight"));
        ly[i].wg = dt_w(g.get(p + "ffn_gate.weight"));
        ly[i].wu = dt_w(g.get(p + "ffn_up.weight"));
        ly[i].wd = dt_w(g.get(p + "ffn_down.weight"));
    }

    float *h, *xb, *q, *att, *proj, *gbuf, *ubuf, *logits, *kc, *vc, *kbuf, *vbuf;
    int *d_token, *d_pos;
    CK(cudaMalloc(&h, n_embd * 4));   CK(cudaMalloc(&xb, n_embd * 4));
    CK(cudaMalloc(&q, n_head * hd * 4)); CK(cudaMalloc(&att, n_head * hd * 4));
    CK(cudaMalloc(&proj, n_embd * 4)); CK(cudaMalloc(&gbuf, n_ff * 4)); CK(cudaMalloc(&ubuf, n_ff * 4));
    CK(cudaMalloc(&logits, n_vocab * 4));
    // int8 KV cache (default on): K/V stored int8 + one fp16 scale per (token,kv_head),
    // so attention reads 4× fewer bytes at depth. KVQ=0 keeps the fp32 cache (the
    // bit-exact gate path). Only the selected layout is allocated.
    bool kvq = getenv("KVQ") ? atoi(getenv("KVQ")) : 0;
    int8_t *kcq = nullptr, *vcq = nullptr; __half *kcs = nullptr, *vcs = nullptr;
    if (kvq) {
        size_t nq = (size_t)n_layer * maxseq * kv_dim, ns = (size_t)n_layer * maxseq * n_kv;
        CK(cudaMalloc(&kcq, nq)); CK(cudaMalloc(&vcq, nq));
        CK(cudaMalloc(&kcs, ns * 2)); CK(cudaMalloc(&vcs, ns * 2));
    } else {
        CK(cudaMalloc(&kc, (size_t)n_layer * maxseq * kv_dim * 4));
        CK(cudaMalloc(&vc, (size_t)n_layer * maxseq * kv_dim * 4));
    }
    CK(cudaMalloc(&kbuf, kv_dim * 4)); CK(cudaMalloc(&vbuf, kv_dim * 4));

    // M12 split-KV attention: ATTN=1 (default) uses the flash-style two-kernel path;
    // ATTN=0 = the monolithic attn_head (the bit-exact gate path). CHUNK = KV positions
    // per block; smaller -> more blocks -> higher occupancy at depth, more combine cost.
    bool attn_split = getenv("ATTN") ? atoi(getenv("ATTN")) : 1;
    int CHUNK = getenv("CHUNK") ? atoi(getenv("CHUNK")) : 192;
    int max_chunks = (maxseq + CHUNK - 1) / CHUNK;
    float *m_part, *l_part, *o_part;
    CK(cudaMalloc(&m_part, (size_t)n_head * max_chunks * 4));
    CK(cudaMalloc(&l_part, (size_t)n_head * max_chunks * 4));
    CK(cudaMalloc(&o_part, (size_t)n_head * max_chunks * hd * 4));
    CK(cudaMalloc(&d_token, 4)); CK(cudaMalloc(&d_pos, 4));
    // M13 multi-step decode: device buffer of produced tokens + a device step counter,
    // so K graph launches run back-to-back without a per-token host round-trip.
    int *d_out, *d_step;
    CK(cudaMalloc(&d_out, ((size_t)ids.size() + n_gen + 8) * 4));
    CK(cudaMalloc(&d_step, 4));
    int8_t* xq8; __half* xd8;                        // quantized-activation scratch
    CK(cudaMalloc(&xq8, n_ff)); CK(cudaMalloc(&xd8, (n_ff / 32) * 2));

    // batched (multi-token) scratch: SPEC_MAX tokens per pass. Used by forward_batch
    // for chunked prefill (and, next, speculative verify).
    float *hB, *xbB, *qB, *attB, *projB, *gB, *uB, *kbufB, *vbufB, *logitsB;
    int8_t* xqB; __half* xdB; int* dpromptB;
    CK(cudaMalloc(&hB,   SPEC_MAX * n_embd * 4));      CK(cudaMalloc(&xbB,  SPEC_MAX * n_embd * 4));
    CK(cudaMalloc(&qB,   SPEC_MAX * n_head * hd * 4)); CK(cudaMalloc(&attB, SPEC_MAX * n_head * hd * 4));
    CK(cudaMalloc(&projB, SPEC_MAX * n_embd * 4));     CK(cudaMalloc(&gB, SPEC_MAX * n_ff * 4));
    CK(cudaMalloc(&uB,   SPEC_MAX * n_ff * 4));        CK(cudaMalloc(&logitsB, (size_t)SPEC_MAX * n_vocab * 4));
    CK(cudaMalloc(&kbufB, SPEC_MAX * kv_dim * 4));     CK(cudaMalloc(&vbufB, SPEC_MAX * kv_dim * 4));
    CK(cudaMalloc(&xqB, SPEC_MAX * n_ff));             CK(cudaMalloc(&xdB, SPEC_MAX * (n_ff / 32) * 2));
    CK(cudaMalloc(&dpromptB, SPEC_MAX * 4));
    int* rdev; CK(cudaMalloc(&rdev, SPEC_MAX * 4));    // per-row argmax for spec verify

    float inv = 1.f / sqrtf((float)hd);
    // Q: quantize activation x[dim] -> shared xq8/xd8.  D: dp4a-GEMV using them
    // (so q/k/v sharing one input are quantized once). GV*: lm_head variants.
    auto Q  = [&](float* x, int dim, cudaStream_t s) { quant_x_q8<<<dim / 32, 32, 0, s>>>(x, xq8, xd8, dim); };
    auto D  = [&](float* y, DTensor W, const float* bias, cudaStream_t s) {
        if (W.type == GT_Q8_0)                                  // requantized weight (e.g. Q4_1 ffn_down)
            gemv_q8_dp4a<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, xq8, xd8, bias);
        else
            gemv_dp4a<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, xq8, xd8, bias);
    };
    auto GV = [&](float* y, DTensor W, const float* x, const float* bias, cudaStream_t s) {
        gemv<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, x, bias);
    };
    // Q6K: int8 dp4a lm_head. Quantize the activation once, then dp4a GEMV.
    auto GVK = [&](float* y, DTensor W, float* x, const float* bias, cudaStream_t s) {
        quant_x_q8<<<W.in / 32, 32, 0, s>>>(x, xq8, xd8, W.in);
        gemv_q6k_dp4a<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, xq8, xd8, bias);
    };
    // Q8 dp4a lm_head (W requantized Q6_K->Q8_0): coalesced, the fast path.
    auto GVK8 = [&](float* y, DTensor W, float* x, const float* bias, cudaStream_t s) {
        quant_x_q8<<<W.in / 32, 32, 0, s>>>(x, xq8, xd8, W.in);
        gemv_q8_dp4a<<<(W.out + 3) / 4, 128, 0, s>>>(y, W, xq8, xd8, bias);
    };

    // batched forward over T (<=SPEC_MAX) tokens `toks` at absolute positions
    // pos0..pos0+T-1. Writes their k,v into the cache and leaves per-token logits in
    // logitsB[T][n_vocab]. Each weight matrix is streamed once for all T tokens.
    auto QB = [&](float* x, int dim, int T, cudaStream_t s) {
        quant_x_q8_batch<<<dim3(dim / 32, T), 32, 0, s>>>(x, xqB, xdB, dim, T);
    };
    auto DB = [&](float* y, DTensor W, const float* bias, int in, int T, cudaStream_t s) {
        if (W.type == GT_Q8_0) launch_q8_dp4a_batch(y, W, xqB, xdB, bias, T, in, s);
        else                   launch_dp4a_batch(y, W, xqB, xdB, bias, T, in, s);
    };
    auto forward_batch = [&](const int* toks, int T, int pos0, cudaStream_t s) {
        embed_q4_batch<<<dim3(1, T), 128, 0, s>>>(hB, tok_emb, toks, T);
        for (int l = 0; l < n_layer; l++) {
            L& y = ly[l];
            rmsnorm_batch<<<T, 256, 0, s>>>(xbB, hB, y.an, n_embd, eps, T);
            QB(xbB, n_embd, T, s);                              // q/k/v share this input
            DB(qB,    y.wq, y.bq, n_embd, T, s);
            DB(kbufB, y.wk, y.bk, n_embd, T, s);
            DB(vbufB, y.wv, y.bv, n_embd, T, s);
            rope_batch<<<dim3((n_head * hd / 2 + 127) / 128, T), 128, 0, s>>>(qB, n_head, hd, pos0, rope_base, n_head * hd);
            rope_batch<<<dim3((kv_dim / 2 + 127) / 128, T), 128, 0, s>>>(kbufB, n_kv, hd, pos0, rope_base, kv_dim);
            if (kvq) {
                int8_t* kcq_l = kcq + (size_t)l * maxseq * kv_dim; __half* kcs_l = kcs + (size_t)l * maxseq * n_kv;
                int8_t* vcq_l = vcq + (size_t)l * maxseq * kv_dim; __half* vcs_l = vcs + (size_t)l * maxseq * n_kv;
                kv_store_batch_q8<<<dim3(n_kv, T), hd, 0, s>>>(kcq_l, kcs_l, vcq_l, vcs_l, kbufB, vbufB, pos0, kv_dim, n_kv, hd, T);
                attn_batch_q8<<<dim3(n_head, T), hd, (hd + maxseq) * 4, s>>>(attB, qB, kcq_l, kcs_l, vcq_l, vcs_l, pos0,
                                                                            n_head, hd, kv_dim, n_kv, gqa, inv, T);
            } else {
                float* kc_l = kc + (size_t)l * maxseq * kv_dim;
                float* vc_l = vc + (size_t)l * maxseq * kv_dim;
                kv_store_batch<<<dim3((kv_dim + 127) / 128, T), 128, 0, s>>>(kc_l, vc_l, kbufB, vbufB, pos0, kv_dim, T);
                attn_batch<<<dim3(n_head, T), hd, (hd + maxseq) * 4, s>>>(attB, qB, kc_l, vc_l, pos0,
                                                                         n_head, hd, kv_dim, gqa, inv, T);
            }
            QB(attB, n_embd, T, s); DB(projB, y.wo, nullptr, n_embd, T, s);
            residual_k<<<(T * n_embd + 127) / 128, 128, 0, s>>>(hB, projB, T * n_embd);
            rmsnorm_batch<<<T, 256, 0, s>>>(xbB, hB, y.fn, n_embd, eps, T);
            QB(xbB, n_embd, T, s); DB(gB, y.wg, nullptr, n_embd, T, s); DB(uB, y.wu, nullptr, n_embd, T, s);
            swiglu_k<<<(T * n_ff + 127) / 128, 128, 0, s>>>(gB, uB, T * n_ff);
            QB(gB, n_ff, T, s); DB(projB, y.wd, nullptr, n_ff, T, s);
            residual_k<<<(T * n_embd + 127) / 128, 128, 0, s>>>(hB, projB, T * n_embd);
        }
        rmsnorm_batch<<<T, 256, 0, s>>>(xbB, hB, out_norm, n_embd, eps, T);
        QB(xbB, n_embd, T, s);                                  // lm_head (Q8_0 verify path)
        if (out_w.type == GT_Q8_0) launch_q8_dp4a_batch(logitsB, out_w, xqB, xdB, nullptr, T, n_embd, s);
        else                       launch_dp4a_batch(logitsB, out_w, xqB, xdB, nullptr, T, n_embd, s);  // q4 lm_head
    };

    // record one full decode step onto stream s. token/pos live in device memory
    // (d_token/d_pos); the step ends by writing the argmax to d_token and pos++,
    // so the captured graph is self-advancing and replayable verbatim.
    auto record = [&](cudaStream_t s) {
        record_token<<<1, 1, 0, s>>>(d_out, d_step, d_token);   // log the input token, bump counter
        embed_q4<<<1, 128, 0, s>>>(h, tok_emb, d_token);
        for (int l = 0; l < n_layer; l++) {
            L& y = ly[l];
            rmsnorm_k<<<1, 256, 0, s>>>(xb, h, y.an, n_embd, eps);
            Q(xb, n_embd, s);                                  // q/k/v share this input
            D(q, y.wq, y.bq, s);
            D(kbuf, y.wk, y.bk, s);
            D(vbuf, y.wv, y.bv, s);
            rope_k<<<(n_head * hd / 2 + 127) / 128, 128, 0, s>>>(q, n_head, hd, d_pos, rope_base);
            rope_k<<<(kv_dim / 2 + 127) / 128, 128, 0, s>>>(kbuf, n_kv, hd, d_pos, rope_base);
            if (kvq) {
                int8_t* kcq_l = kcq + (size_t)l * maxseq * kv_dim; __half* kcs_l = kcs + (size_t)l * maxseq * n_kv;
                int8_t* vcq_l = vcq + (size_t)l * maxseq * kv_dim; __half* vcs_l = vcs + (size_t)l * maxseq * n_kv;
                kv_store_q8<<<n_kv, hd, 0, s>>>(kcq_l, kcs_l, vcq_l, vcs_l, kbuf, vbuf, d_pos, kv_dim, n_kv, hd);
                attn_head_q8<<<n_head, hd, (hd + maxseq) * 4, s>>>(att, q, kcq_l, kcs_l, vcq_l, vcs_l, d_pos,
                                                                  n_head, hd, kv_dim, n_kv, gqa, inv);
            } else {
                float* kc_l = kc + (size_t)l * maxseq * kv_dim;
                float* vc_l = vc + (size_t)l * maxseq * kv_dim;
                kv_store<<<(kv_dim + 127) / 128, 128, 0, s>>>(kc_l, vc_l, kbuf, vbuf, d_pos, kv_dim);
                if (attn_split) {                          // M12: flash-style split-KV
                    attn_partial<<<dim3(n_head, max_chunks), hd, (hd + CHUNK) * 4, s>>>(
                        q, kc_l, vc_l, d_pos, m_part, l_part, o_part, n_head, hd, kv_dim, gqa, inv, CHUNK, max_chunks);
                    attn_combine<<<n_head, hd, 0, s>>>(att, m_part, l_part, o_part, d_pos, n_head, hd, CHUNK, max_chunks);
                } else {
                    attn_head<<<n_head, hd, (hd + maxseq) * 4, s>>>(att, q, kc_l, vc_l, d_pos,
                                                                    n_head, hd, kv_dim, gqa, inv);
                }
            }
            Q(att, n_embd, s); D(proj, y.wo, nullptr, s);
            residual_k<<<(n_embd + 127) / 128, 128, 0, s>>>(h, proj, n_embd);
            rmsnorm_k<<<1, 256, 0, s>>>(xb, h, y.fn, n_embd, eps);
            Q(xb, n_embd, s);                                  // gate/up share this input
            D(gbuf, y.wg, nullptr, s);
            D(ubuf, y.wu, nullptr, s);
            swiglu_k<<<(n_ff + 127) / 128, 128, 0, s>>>(gbuf, ubuf, n_ff);
            Q(gbuf, n_ff, s); D(proj, y.wd, nullptr, s);
            residual_k<<<(n_embd + 127) / 128, 128, 0, s>>>(h, proj, n_embd);
        }
        rmsnorm_k<<<1, 256, 0, s>>>(xb, h, out_norm, n_embd, eps);
        switch (lmhead) {                                       // lm_head Q6_K variant
            case 0: GV (logits, out_w, xb, nullptr, s); break;  // original scalar deq
            case 1: gemv_q6k_fp32<<<(out_w.out + 3) / 4, 128, 0, s>>>(logits, out_w, xb, nullptr); break;
            case 2: GVK (logits, out_w, xb, nullptr, s); break;  // Q6_K int8 dp4a
            case 3: GVK8(logits, out_w, xb, nullptr, s); break;  // Q8_0 int8 dp4a (fast)
            case 4: quant_x_q8<<<out_w.in / 32, 32, 0, s>>>(xb, xq8, xd8, out_w.in);  // q4_0: ½ the bytes
                    gemv_dp4a<<<(out_w.out + 3) / 4, 128, 0, s>>>(logits, out_w, xq8, xd8, nullptr); break;
        }
        argmax_k<<<1, 256, 0, s>>>(d_token, logits, n_vocab);
        advance_k<<<1, 1, 0, s>>>(d_pos);
    };

    fprintf(stderr, "minfer-cuda: qwen2 L=%d d=%d ff=%d H=%d/%d hd=%d vocab=%d maxseq=%d\n",
            n_layer, n_embd, n_ff, n_head, n_kv, hd, n_vocab, maxseq);

    // capture the decode step into a CUDA graph (one cudaGraphLaunch / token)
    cudaStream_t s; CK(cudaStreamCreate(&s));
    // Two decode graphs: monolithic attention (best at short context) and split-KV
    // (M12, wins at depth). Dispatch by pos at launch — no short-context regression,
    // depth win for free. Split only differs when kvq==0 (the int8 path keeps attn_head_q8).
    cudaGraph_t graph; cudaGraphExec_t exec_m, exec_s = nullptr;
    bool split_master = attn_split && !kvq;
    attn_split = false;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    record(s);
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&exec_m, graph, 0));
    if (split_master) {
        attn_split = true;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        record(s);
        CK(cudaStreamEndCapture(s, &graph));
        CK(cudaGraphInstantiate(&exec_s, graph, 0));
    }
    int ATTNX = getenv("ATTNX") ? atoi(getenv("ATTNX")) : 256;  // pos crossover mono->split
    auto exec_for = [&](int pos) { return (exec_s && pos >= ATTNX) ? exec_s : exec_m; };
    cudaGraphExec_t exec = exec_m;                               // default for pos-agnostic callers

    // microbench: cost of one batched forward vs T. Reveals whether verify-of-K is
    // ~free (memory-bound) or ~K× (compute-bound) on this int8 engine.  BENCHT=1
    if (getenv("BENCHT")) {
        int dummy[SPEC_MAX]; for (int i = 0; i < SPEC_MAX; i++) dummy[i] = ids[i % ids.size()];
        CK(cudaMemcpy(dpromptB, dummy, SPEC_MAX * 4, cudaMemcpyHostToDevice));
        for (int T = 1; T <= SPEC_MAX; T <<= 1) {
            for (int w = 0; w < 3; w++) forward_batch(dpromptB, T, 0, s);   // warmup
            CK(cudaStreamSynchronize(s));
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            int N = 30; cudaEventRecord(a, s);
            for (int it = 0; it < N; it++) forward_batch(dpromptB, T, 0, s);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= N;
            fprintf(stderr, "forward_batch T=%d : %.2f ms/pass  (%.2f ms/token, %.2fx vs T=1-equiv)\n",
                    T, ms, ms / T, ms);
        }
        return 0;
    }
    int zero = 0; CK(cudaMemcpy(d_pos, &zero, 4, cudaMemcpyHostToDevice));
    int use_batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 0;   // batched prefill (needs LMHEAD=3)
    cudaEvent_t p0, p1; cudaEventCreate(&p0); cudaEventCreate(&p1);
    cudaEventRecord(p0, s);
    if (use_batch) {
        // prefill in chunks of <=SPEC_MAX through forward_batch (one weight stream
        // per chunk). The last chunk's last row predicts the first generated token.
        int P = (int)ids.size(), pos0 = 0, lastT = 0;
        while (pos0 < P) {
            int T = std::min(SPEC_MAX, P - pos0);
            CK(cudaMemcpyAsync(dpromptB, ids.data() + pos0, T * 4, cudaMemcpyHostToDevice, s));
            forward_batch(dpromptB, T, pos0, s);
            pos0 += T; lastT = T;
        }
        argmax_k<<<1, 256, 0, s>>>(d_token, logitsB + (size_t)(lastT - 1) * n_vocab, n_vocab);
        CK(cudaMemcpy(d_pos, &P, 4, cudaMemcpyHostToDevice));
    } else {
        // prefill: override d_token with each prompt token (the graph's own argmax
        // write is discarded until the last prompt token, which yields token #1).
        for (size_t i = 0; i < ids.size(); i++) {
            CK(cudaMemcpyAsync(d_token, &ids[i], 4, cudaMemcpyHostToDevice, s));
            CK(cudaGraphLaunch(exec_for((int)i), s));          // pos i
        }
    }
    cudaEventRecord(p1, s); cudaEventSynchronize(p1);
    float pms = 0; cudaEventElapsedTime(&pms, p0, p1);
    fprintf(stderr, "minfer-cuda prefill: %zu tok in %.1f ms (%s)\n",
            ids.size(), pms, use_batch ? "batched" : "sequential");

    int use_spec = getenv("SPEC") ? atoi(getenv("SPEC")) : 0;     // n-gram speculative decode
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    if (!use_spec) {
        // M13 multi-step baseline generation: launch STEP self-advancing graphs
        // back-to-back (each logs its input token to d_out via record_token), then one
        // D2H copy per chunk. Cuts host syncs ~STEP×, removing the per-token GPU idle
        // gap. MSTEP=1 == the old per-token path (for A/B). The graph self-chains on
        // device (d_token/d_pos), so no host involvement is needed between launches.
        // Default 1 (smooth streaming): multi-step is a measured WASH at full clock —
        // decode is memory-bandwidth-bound, so removing the per-token host sync (which is
        // ~0.3% of a step) does nothing. MSTEP>1 kept as an opt-in / experiment knob.
        int STEP = getenv("MSTEP") ? atoi(getenv("MSTEP")) : 1;
        if (STEP < 1) STEP = 1;
        cudaEventRecord(t0, s);
        std::vector<int> hbuf(STEP);
        int emitted = 0; bool done = false;
        while (emitted < n_gen && !done) {
            int k = std::min(STEP, n_gen - emitted);
            CK(cudaMemsetAsync(d_step, 0, 4, s));                 // chunk-local index 0..k-1
            for (int j = 0; j < k; j++)
                CK(cudaGraphLaunch(exec_for((int)ids.size() + emitted + j), s));
            CK(cudaMemcpy(hbuf.data(), d_out, k * 4, cudaMemcpyDeviceToHost));  // one sync / chunk
            for (int j = 0; j < k; j++) {
                int tok = hbuf[j];
                if (!raw_ids && tok == eos_id) { done = true; break; }  // clean stop in text mode
                emit(tok); emitted++;
            }
        }
        cudaEventRecord(t1, s); cudaEventSynchronize(t1);
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
        printf("\n");
        fprintf(stderr, "minfer-cuda decode: %d tok in %.1f ms = %.2f tok/s\n", emitted, ms, emitted * 1000.f / ms);
        return 0;
    }

    // ---- n-gram (prompt-lookup) speculative decode ----
    // Lossless: a drafted token is committed only if it equals the true greedy
    // argmax, so output is token-for-token identical to baseline. Draft = the
    // tokens that followed the last earlier occurrence of the current suffix.
    int NG     = getenv("NG")     ? atoi(getenv("NG"))     : 3;   // n-gram suffix length
    int DRAFTK = getenv("DRAFTK") ? atoi(getenv("DRAFTK")) : 4;   // max drafted tokens / cycle
    DRAFTK = std::min(DRAFTK, SPEC_MAX - 1);
    if (lmhead != 3 && lmhead != 4) { fprintf(stderr, "SPEC needs LMHEAD=3 or 4 (batched lm_head)\n"); return 1; }

    auto ngram_draft = [&](const std::vector<int>& seq, int K, int* draft) -> int {
        int L = (int)seq.size();
        if (L < NG + 1 || K <= 0) return 0;
        for (int j = L - NG - 1; j >= 0; j--) {            // latest earlier match wins
            bool m = true;
            for (int k = 0; k < NG; k++) if (seq[j + k] != seq[L - NG + k]) { m = false; break; }
            if (m) {
                int avail = L - (j + NG), n = std::min(K, avail);
                for (int k = 0; k < n; k++) draft[k] = seq[j + NG + k];
                return n;
            }
        }
        return 0;
    };

    std::vector<int> seq = ids;                           // confirmed tokens, positions 0..pos
    int cur; CK(cudaMemcpy(&cur, d_token, 4, cudaMemcpyDeviceToHost));  // token at position P
    int pos = (int)ids.size();
    seq.push_back(cur);                                   // seq holds positions 0..pos
    int draft[SPEC_MAX], hb[SPEC_MAX], r[SPEC_MAX];
    long n_emit = 0, n_cycles = 0, n_drafted = 0, n_accepted = 0, n_specruns = 0;
    bool dev_synced = true;                               // d_token==cur && d_pos==pos on device?
    // Adaptive gate: verify is costly, so only speculate while it pays off. ewma
    // tracks recent accept fraction; below GATE we fall back to plain decode but
    // probe occasionally (small K) to re-detect repetition. Makes SPEC never regress.
    float ewma = 0.6f; const float GATE = 0.55f; long probe = 0;

    cudaEventRecord(t0, s);
    while (n_emit < n_gen) {
        bool allow = ewma >= GATE, probing = false;
        if (!allow && (probe++ % 32) == 0) { allow = true; probing = true; }
        int K = allow ? ngram_draft(seq, DRAFTK, draft) : 0;
        if (probing) K = std::min(K, 1);                 // cheap probe (verify 2 tokens)
        K = std::min(K, maxseq - 1 - pos);               // KV headroom

        if (K == 0) {                                    // no draft -> fast graph step (no regression)
            if (!dev_synced) {                           // only resync after a spec run; the graph
                CK(cudaMemcpyAsync(d_token, &cur, 4, cudaMemcpyHostToDevice, s));  // self-advances so
                CK(cudaMemcpyAsync(d_pos, &pos, 4, cudaMemcpyHostToDevice, s));    // runs of single
            }                                            // steps need no uploads (== baseline cost)
            CK(cudaGraphLaunch(exec_for(pos), s));        // processes cur@pos, writes KV[pos], emits next
            int nxt; CK(cudaMemcpy(&nxt, d_token, 4, cudaMemcpyDeviceToHost));
            if (!raw_ids && cur == eos_id) break;        // clean stop in text mode
            emit(cur); n_emit++;
            cur = nxt; pos++; seq.push_back(cur);
            dev_synced = true;
            n_cycles++;
            continue;
        }

        int T = K + 1;                                   // verify [cur, draft0..draftK-1] @ pos..pos+K
        hb[0] = cur; for (int k = 0; k < K; k++) hb[1 + k] = draft[k];
        CK(cudaMemcpyAsync(dpromptB, hb, T * 4, cudaMemcpyHostToDevice, s));
        forward_batch(dpromptB, T, pos, s);
        for (int j = 0; j < T; j++)                       // r[j] = true token at position pos+j+1
            argmax_k<<<1, 256, 0, s>>>(rdev + j, logitsB + (size_t)j * n_vocab, n_vocab);
        CK(cudaMemcpy(r, rdev, T * 4, cudaMemcpyDeviceToHost));

        int a = 0; while (a < K && draft[a] == r[a]) a++;  // longest verified prefix
        bool stop = false;
        if (!raw_ids && cur == eos_id) break;            // clean stop in text mode
        emit(cur); n_emit++;                             // cur (token @pos) is confirmed
        for (int k = 0; k < a && n_emit < n_gen; k++) {
            if (!raw_ids && draft[k] == eos_id) { stop = true; break; }
            emit(draft[k]); n_emit++;
        }
        if (stop) break;
        for (int k = 0; k < a; k++) seq.push_back(draft[k]);
        cur = r[a]; pos += a + 1; seq.push_back(cur);     // r[a] = correction (true token @pos+a+1)
        dev_synced = false;                               // forward_batch left d_token/d_pos stale
        ewma = 0.7f * ewma + 0.3f * ((float)a / K);       // update acceptance estimate (fast)
        n_cycles++; n_specruns++; n_drafted += K; n_accepted += a;
    }
    cudaEventRecord(t1, s); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("\n");
    fprintf(stderr, "minfer-cuda SPEC decode: %ld tok in %.1f ms = %.2f tok/s | "
            "cycles=%ld specruns=%ld accept=%.0f%% (%ld/%ld) NG=%d K=%d\n",
            n_emit, ms, n_emit * 1000.f / ms, n_cycles, n_specruns,
            n_drafted ? 100.0 * n_accepted / n_drafted : 0.0, n_accepted, n_drafted, NG, DRAFTK);
    return 0;
}
