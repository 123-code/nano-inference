// minfer M15-M18 — the whole decode loop as ONE persistent kernel.
//
// cuda_infer runs a token as a CUDA graph of 567 kernels. Measured (bench/breakdown.cu):
// its GEMVs stream at 75 GB/s against a 101 GB/s roofline (warp-per-row wastes lanes on
// 1536-wide rows: gate/up 75 GB/s, k/v 39 GB/s), and the 368 tiny kernels leave DRAM idle
// for 1.8 ms of a 13.3 ms step. Here one cooperative kernel (1 CTA per SM) runs every
// layer of every token:
//   - weights are repacked at load into 32-row groups: a warp reads 512 contiguous bytes
//     per load and each lane owns one row, so there is no cross-lane reduction;
//   - the activation vector is quantized once into shared memory (broadcast reads);
//   - phases are separated by a ~1.2 us grid barrier instead of kernel launches;
//   - argmax, the next token's embedding and the token loop stay on the GPU; the host
//     just drains a mapped ring buffer.
// M16: a step can carry T <= 8 tokens (prompt chunks, speculative verify). T = 1 keeps the dp4a
// path; T > 1 runs the same packed weights through tensor cores (mma.m16n8k32 s8), so one weight
// pass serves all T tokens. Both paths form identical per-block integer sums and accumulate them
// through the same explicitly rounded expression in the same order, so a token's logits do not
// depend on T. M17: n-gram speculative decoding on the GPU, lossless by that property.
// M18: Qwen2.5 0.5B / 1.5B / 3B — the model shape is a compile-time template parameter; row groups
// split unevenly across CTAs, per-matrix chunk size and split-K, q/k rows permuted at load so
// RoPE pairs are adjacent rows (any 32-row split keeps them together; q.k is unchanged).
// Numerics follow cuda_infer (q4_0 x int8 activations, same quantization, fp32 KV cache);
// only float summation order differs.
//
//   mega <model.gguf> <n_gen> <id0> <id1> ...      env: RAWIDS=1 prints token ids
//   TCHUNK=<T>  prompt tokens per step (default 8)
//   SPEC=0      disable n-gram speculative decoding (NGMAX=3 longest n-gram, SPEC_SLOPE=0.074 cost model)
//   TEACHER=<k> ids = a reference sequence whose first k are the prompt; predict every
//               position under teacher forcing and report top-1 agreement + margins.
#include "gguf.h"
#include "detok.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace minfer;

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "mega: unsupported shape: %s\n", #c); exit(1); } } while (0)

constexpr int NCTA = 8, NT = 512, NW = NT / 32;     // one CTA per SM of the Orin Nano
constexpr int TMAX = 8;                             // tokens per step (one mma n-tile)
constexpr unsigned FULL = 0xffffffffu;

// model shape, compile-time: lets loops over heads/dims unroll and keeps registers in check
template<int E_, int FF_, int NH_, int NKV_, int HD_> struct Shape {
    static constexpr int E = E_, FF = FF_, NH = NH_, NKV = NKV_, HD = HD_;
    static constexpr int G = NH / NKV, KVD = NKV * HD, QKV = (NH + 2 * NKV) * HD;
    static constexpr int XE = E / 4 + 4, XF = FF / 4 + 4;   // smem words per token (+4: mma B loads conflict-free)
    static constexpr int TS = FF / 32;                       // per-token stride of block scales
    static_assert(E == NH * HD && HD % 32 == 0 && E % 32 == 0 && FF % 32 == 0 && QKV % 32 == 0, "shape");
};
using Qwen05 = Shape<896, 4864, 14, 2, 64>;     // Qwen2.5-0.5B
using Qwen15 = Shape<1536, 8960, 12, 2, 128>;   // Qwen2.5-1.5B
using Qwen3  = Shape<2048, 11008, 16, 2, 128>;  // Qwen2.5-3B

// q4_0 matrix packed in 32-row groups: lane r of a warp owns row g*32+r. Blocks are processed in
// chunks of cb (8, or 4 when the row length is not a multiple of 256), split-K into ks parts.
//   q[(g*nb + b)*32 + r]          = the 16 nibble bytes of row g*32+r, block b
//   s[((g*nc + c)*32 + r)*cb]     = that row's cb fp16 scales for blocks c*cb .. c*cb+cb-1
struct PM { const uint4* q; const __half* s; int nb, cb, ks; };
struct Layer { PM qkv, o, g, u, d; const float *an, *fn, *bqkv; };
struct Cand { float best, second; int idx, pad; };
struct Cfg {
    int NL, maxseq, P, eos, stop_eos, teacher, prof, tchunk, spec, ngmax;
    float spec_slope;                       // cost of each extra token in a step, relative to a 1-token step
    int glm, ts;                            // lm_head row groups; per-token stride of block scales
    int off_xd, off_xs8, off_hb, off_red, off_wts, off_cand, off_scale, off_hist;   // dynamic smem layout (bytes)
    float eps, rope_base, inv_sqrt_hd;
    PM lm; const float* out_norm; const uint8_t* emb; int emb_q8;   // emb: raw Q4_0 (18 B) or Q8_0 (34 B) blocks
    float *h, *q, *am, *al, *ao, *act_d, *kc, *vc;      // per-token buffers are [TMAX][...]
    int8_t* act_q; int* act_s8; Cand* cand; const int* prompt;
    int *r_count, *r_tok; long long* r_ns; float* r_margin;   // host-mapped ring
    long long* stats;                       // [gen steps, drafted, accepted]
};
__constant__ Cfg C;
__constant__ Layer LY[64];

// dynamic shared memory: [ union: int8 activations (T tokens) + split partials | attention
// scratch ] [ xd | xs8 | hb | red | wts | cand | scale ] — offsets computed on the host. Views are
// derived from the __shared__ symbol itself (not passed as pointers), so even out-of-line functions
// address it with LDS/STS instead of generic loads.
extern __shared__ __align__(16) unsigned char smem[];
struct Smem {
    __device__ static unsigned char* uni() { return smem; }
    __device__ static int* xq() { return reinterpret_cast<int*>(smem); }
    __device__ static float* part(int xs) { return reinterpret_cast<float*>(smem + (size_t)xs * TMAX * 4); }
    // E-wide token vectors [t][E] staged for quant_tokens; shares space with part(XE), which is
    // only written after the inputs have been quantized
    __device__ static float* xs(int xe) { return part(xe); }
    __device__ static float* xd() { return reinterpret_cast<float*>(smem + C.off_xd); }
    __device__ static int* xs8() { return reinterpret_cast<int*>(smem + C.off_xs8); }
    __device__ static float* hb() { return reinterpret_cast<float*>(smem + C.off_hb); }
    __device__ static float* red() { return reinterpret_cast<float*>(smem + C.off_red); }
    __device__ static float* wts() { return reinterpret_cast<float*>(smem + C.off_wts); }
    __device__ static Cand* cand() { return reinterpret_cast<Cand*>(smem + C.off_cand); }
    __device__ static float* scale() { return reinterpret_cast<float*>(smem + C.off_scale); }
    __device__ static int* hist() { return reinterpret_cast<int*>(smem + C.off_hist); }   // token at each position
};

// ---------------- primitives ----------------
__device__ unsigned g_bar_count, g_bar_gen;
__device__ unsigned long long g_prof[32], g_clk[4];   // g_clk: clock64/globaltimer at kernel start/end   // PROFILE=1: ns per phase on CTA 0 (critical path)
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// close the current phase k: charge the time since the last mark to g_prof[k]
__device__ __forceinline__ void mark(unsigned long long& t, int k) {
    if (C.prof && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long n = gtime(); g_prof[k] += n - t; t = n; }
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v; asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
// sense-reversing grid barrier (all CTAs co-resident: cooperative launch)
__device__ __forceinline__ void grid_sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned gen = ld_acquire(&g_bar_gen);
        if (atomicAdd(&g_bar_count, 1u) == gridDim.x - 1) {
            atomicExch(&g_bar_count, 0u);
            st_release(&g_bar_gen, gen + 1);
        } else {
            while (ld_acquire(&g_bar_gen) == gen) {}
        }
        __threadfence();
    }
    __syncthreads();
}
// weights are immutable for the kernel's lifetime: skip L1, ask L2 for 256 B lines
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned ldw32(const unsigned* p) {
    unsigned v; asm("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
// L2-coherent loads (KV is written by other SMs inside this kernel), as asm volatile so a batch of
// them is issued back to back instead of being sunk next to its uses
__device__ __forceinline__ float ld_cg(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ float4 ld_cg4(const float4* p) {
    float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v;
}
// one q4_0 block's contribution: acc + (dw*dx) * float(sum(n*x) - 8*sum(x)). Explicit rounding
// so the dp4a (T = 1) and tensor-core (T > 1) paths produce bit-identical results.
__device__ __forceinline__ float blk_acc(float acc, float dw, float dx, int isum, int s8) {
    return __fmaf_rn(__fmul_rn(dw, dx), __int2float_rn(isum - s8), acc);
}
// q4_0 block x int8 activations: same SIMD-nibble dp4a sum as cuda_infer's gemv_dp4a
__device__ __forceinline__ int dot_q4(uint4 p, const int* x) {
    const int4 xa = *reinterpret_cast<const int4*>(x), xc = *reinterpret_cast<const int4*>(x + 4);
    const unsigned M = 0x0F0F0F0Fu;
    int a = __dp4a((int)(p.x & M), xa.x, 0);
    a = __dp4a((int)(p.y & M), xa.y, a); a = __dp4a((int)(p.z & M), xa.z, a); a = __dp4a((int)(p.w & M), xa.w, a);
    a = __dp4a((int)((p.x >> 4) & M), xc.x, a); a = __dp4a((int)((p.y >> 4) & M), xc.y, a);
    a = __dp4a((int)((p.z >> 4) & M), xc.z, a); a = __dp4a((int)((p.w >> 4) & M), xc.w, a);
    return a;
}
__device__ __forceinline__ uint2 ldw8(const uint2* p) {
    uint2 v; asm("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v;
}
// the CB fp16 scales of one (group, chunk, row), as CB/2 half2
template<int CB> __device__ __forceinline__ void load_scales(const __half* p, __half2 (&h)[CB / 2]) {
    if constexpr (CB == 8) { const uint4 v = ldw(reinterpret_cast<const uint4*>(p)); memcpy(h, &v, 16); }
    else { const uint2 v = ldw8(reinterpret_cast<const uint2*>(p)); memcpy(h, &v, 8); }
}
template<int CB> __device__ __forceinline__ float scale_of(const __half2 (&h)[CB / 2], int k) {
    return (k & 1) ? __high2float(h[k >> 1]) : __low2float(h[k >> 1]);
}
// T = 1: lane's row of group g over block chunks [c0, c1) (CB blocks each), token 0 of xq/xd/xs8
template<int CB>
__device__ __forceinline__ float row_dot(const PM& M, int g, int c0, int c1, const int* xq, const float* xd, const int* xs8, int lane) {
    const int nc = M.nb / CB;
    const uint4* qp = M.q + ((size_t)g * M.nb + c0 * CB) * 32 + lane;
    const __half* sp = M.s + (((size_t)g * nc + c0) * 32 + lane) * CB;
    float acc = 0.f;
    for (int c = c0; c < c1; c++, qp += 32 * CB, sp += 32 * CB) {
        uint4 w[CB];
        #pragma unroll
        for (int k = 0; k < CB; k++) w[k] = ldw(qp + k * 32);
        __half2 sh[CB / 2]; load_scales<CB>(sp, sh);
        #pragma unroll
        for (int k = 0; k < CB; k++) {
            const int b = c * CB + k;
            acc = blk_acc(acc, scale_of<CB>(sh, k), xd[b], dot_q4(w[k], xq + b * 8), xs8[b]);
        }
    }
    return acc;
}
__device__ __forceinline__ void mma_s8(int (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    d[0] = d[1] = d[2] = d[3] = 0;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// T > 1: the warp computes group g's 32 rows x 8 token columns over chunks [c0, c1) with
// mma.m16n8k32 (s8). Lane (gid = lane/4, tig = lane%4) owns rows gid + 8i (i = 0..3) x tokens
// 2tig, 2tig+1 in acc[i][j]. A fragment = the nibbles of word tig of each row (low nibbles =
// elements 4tig.., high = 16+4tig..: q4_0's own order), B fragment = token gid's int8 words.
// xs: smem words per token of xq; ts: per-token stride of xd / xs8.
template<int CB>
__device__ __forceinline__ void tile_dot(const PM& M, int g, int c0, int c1, const int* xq, int xs,
                                         const float* xd, const int* xs8, int ts, int lane, float (&acc)[4][2]) {
    const int gid = lane >> 2, tig = lane & 3, nc = M.nb / CB;
    const unsigned* qw = reinterpret_cast<const unsigned*>(M.q + (size_t)g * M.nb * 32) + gid * 4 + tig;
    const __half* sp = M.s + ((size_t)g * nc * 32 + gid) * CB;
    const int* xb = xq + gid * xs + tig;
    const float *xd0 = xd + 2 * tig * ts, *xd1 = xd0 + ts;
    const int *s80 = xs8 + 2 * tig * ts, *s81 = s80 + ts;
    const unsigned MK = 0x0F0F0F0Fu;
    for (int c = c0; c < c1; c++) {
        __half2 sh[4][CB / 2];
        #pragma unroll
        for (int i = 0; i < 4; i++) load_scales<CB>(sp + ((size_t)c * 32 + 8 * i) * CB, sh[i]);
        #pragma unroll
        for (int k = 0; k < CB; k++) {
            const int b = c * CB + k;
            unsigned w[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) w[i] = ldw32(qw + ((size_t)b * 32 + 8 * i) * 4);
            const unsigned b0 = xb[b * 8], b1 = xb[b * 8 + 4];
            int d0[4], d1[4];
            mma_s8(d0, w[0] & MK, w[1] & MK, (w[0] >> 4) & MK, (w[1] >> 4) & MK, b0, b1);   // rows gid, gid+8
            mma_s8(d1, w[2] & MK, w[3] & MK, (w[2] >> 4) & MK, (w[3] >> 4) & MK, b0, b1);   // rows gid+16, gid+24
            const float dx0 = xd0[b], dx1 = xd1[b];
            const int s0 = s80[b], s1 = s81[b];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float dw = scale_of<CB>(sh[i], k);
                const int* d = i < 2 ? d0 : d1;
                const int e = (i & 1) * 2;          // d[e], d[e+1] = (row gid + 8i, tokens 2tig, 2tig+1)
                acc[i][0] = blk_acc(acc[i][0], dw, dx0, d[e], s0);
                acc[i][1] = blk_acc(acc[i][1], dw, dx1, d[e + 1], s1);
            }
        }
    }
}
// split-K GEMV over row groups [g0, g0+ng): task = (group, split), task t runs on warp
// (t0 + t) % NW so consecutive calls keep the warps balanced. Split sp covers chunks
// [sp*nc/ks, (sp+1)*nc/ks). Output at float offset poff of the smem union:
//   T = 1: part[split*ng*32 + row]                 T > 1: part[(split*TMAX + tok)*ng*32 + row]
template<int CB>
__device__ __forceinline__ void gemv_part_t(const PM& M, int g0, int ng, int poff, int t0, int T, int xs, int warp, int lane) {
    float* part = reinterpret_cast<float*>(Smem::uni()) + poff;
    const int nc = M.nb / CB, ks = M.ks, ntask = ng * ks, gid = lane >> 2, tig = lane & 3;
    for (int t = ((warp - t0) % NW + NW) % NW; t < ntask; t += NW) {
        const int gl = t / ks, sp = t % ks, c0 = sp * nc / ks, c1 = (sp + 1) * nc / ks;
        if (T == 1) {
            part[sp * ng * 32 + gl * 32 + lane] = row_dot<CB>(M, g0 + gl, c0, c1, Smem::xq(), Smem::xd(), Smem::xs8(), lane);
            continue;
        }
        float acc[4][2] = {};
        tile_dot<CB>(M, g0 + gl, c0, c1, Smem::xq(), xs, Smem::xd(), Smem::xs8(), C.ts, lane, acc);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                if (2 * tig + j < T) part[(sp * TMAX + 2 * tig + j) * ng * 32 + gl * 32 + gid + 8 * i] = acc[i][j];
    }
}
__device__ __noinline__ void gemv_part(const PM& M, int g0, int ng, int poff, int t0, int T, int xs, int warp, int lane) {
    if (M.cb == 8) gemv_part_t<8>(M, g0, ng, poff, t0, T, xs, warp, lane);
    else           gemv_part_t<4>(M, g0, ng, poff, t0, T, xs, warp, lane);
}
// T > 1 for the gate/up phase: a warp runs every split of its group and adds them in split
// order — the same sum the T = 1 path forms from part[] — so only the totals need smem.
// out[tok*ng*32 + row] at float offset poff of the smem union
template<int CB>
__device__ __forceinline__ void gemv_seq_t(const PM& M, int g0, int ng, int poff, int t0, int T, int xs, int warp, int lane) {
    float* out = reinterpret_cast<float*>(Smem::uni()) + poff;
    const int nc = M.nb / CB, ks = M.ks, gid = lane >> 2, tig = lane & 3;
    for (int t = ((warp - t0) % NW + NW) % NW; t < ng; t += NW) {
        float tot[4][2];
        for (int sp = 0; sp < ks; sp++) {
            float acc[4][2] = {};
            tile_dot<CB>(M, g0 + t, sp * nc / ks, (sp + 1) * nc / ks, Smem::xq(), xs, Smem::xd(), Smem::xs8(), C.ts, lane, acc);
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++) tot[i][j] = sp ? __fadd_rn(tot[i][j], acc[i][j]) : acc[i][j];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                if (2 * tig + j < T) out[(2 * tig + j) * ng * 32 + t * 32 + gid + 8 * i] = tot[i][j];
    }
}
__device__ __noinline__ void gemv_seq(const PM& M, int g0, int ng, int poff, int t0, int T, int xs, int warp, int lane) {
    if (M.cb == 8) gemv_seq_t<8>(M, g0, ng, poff, t0, T, xs, warp, lane);
    else           gemv_seq_t<4>(M, g0, ng, poff, t0, T, xs, warp, lane);
}
// symmetric int8 per 32-block, exactly cuda_infer's quant_x_q8 (lane = element)
// (the shuffles run even when !own, so callers can keep a block-uniform trip count)
__device__ __forceinline__ void quant32(float v, int b, int8_t* xq, float* xd, int* xs8, int lane, bool own = true) {
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
    const float inv = (a > 0.f) ? 127.f / a : 0.f;
    const int q = __float2int_rn(v * inv);
    if (own) xq[b * 32 + lane] = (int8_t)q;
    int s = q;
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (own && lane == 0) { xd[b] = __half2float(__float2half(a / 127.f)); xs8[b] = 8 * s; }
}
// token vectors X[t][E] (t < T, staged in smem) -> optionally rmsnorm(x_t) * w -> int8 blocks of
// each token in smem. Every loop that holds a shuffle has a block-uniform trip count so the warp
// stays provably converged; each token gets exactly the arithmetic of a lone token.
template<class M>
__device__ __forceinline__ void quant_tokens(const float* X, const float* w, int T) {
    constexpr int n = M::E, nb = n / 32, TS = M::TS, XE = M::XE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (w) {
        for (int t = 0; t < T; t++) {
            float ss = 0.f;
            for (int i = tid; i < n; i += NT) ss = __fmaf_rn(X[t * n + i], X[t * n + i], ss);
            #pragma unroll
            for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(FULL, ss, o));
            if (lane == 0) Smem::red()[t * NW + warp] = ss;
        }
        __syncthreads();
        if (tid < T) {
            float s = 0.f;
            #pragma unroll
            for (int v = 0; v < NW; v++) s = __fadd_rn(s, Smem::red()[tid * NW + v]);
            Smem::scale()[tid] = rsqrtf(__fadd_rn(__fdiv_rn(s, (float)n), C.eps));
        }
        __syncthreads();
    }
    for (int r = 0; r < (T * nb + NW - 1) / NW; r++) {  // (token, block) pairs, uniform trip count
        const int i0 = warp + r * NW, i = min(i0, T * nb - 1), t = i / nb, b = i % nb, e = b * 32 + lane;
        const float v = w ? __fmul_rn(__fmul_rn(X[t * n + e], Smem::scale()[t]), __ldg(w + e)) : X[t * n + e];
        quant32(v, b, reinterpret_cast<int8_t*>(Smem::xq() + t * XE), Smem::xd() + t * TS, Smem::xs8() + t * TS, lane, i0 < T * nb);
    }
    __syncthreads();
}
template<class M> __device__ __forceinline__ void load_tokens(float* X, const float* src, int T) {
    for (int i = threadIdx.x; i < T * M::E; i += NT) X[i] = __ldcg(src + i);
    __syncthreads();
}
// keep the larger (lower index on ties); second = best of everything else
__device__ __forceinline__ void cand_merge(float& b, int& i, float& s, float ob, int oi, float os) {
    const bool take = ob > b || (ob == b && oi < i);
    s = fmaxf(fmaxf(s, os), take ? b : ob);
    if (take) { b = ob; i = oi; }
}
__device__ __forceinline__ void cand_push(float& b, int& i, float& s, float v, int row) {   // rows arrive in increasing order
    if (v > b) { s = b; b = v; i = row; } else if (v > s) s = v;
}

// ---------------- attention ----------------
// CTA c owns kv head c/NCH and position chunk c%NCH of every query's causal range. For each
// token of the step (one after another) all 16 warps split that chunk, each running online softmax
// for the G q heads of the kv head at once (one K/V read serves every head; lane = HD/32 dims), PB
// positions per batch: scores are kept in log2 units (q pre-scaled by log2(e)/sqrt(hd)) and the
// running state is rescaled once per batch. All 16 warp states then go to smem and warp j folds
// head j's states in warp order into partial [t][head][chunk] (m in log2 units). The split and the
// fold order depend only on the token's position, so a token's numbers do not depend on how many
// tokens share the step. Loops that hold shuffles run a block-uniform trip count and branch only
// around shuffle-free code, so the warp stays provably converged (otherwise every SHFL becomes a
// call that serializes the heads' reductions). G and HD are compile-time so the heads' chains
// interleave.
template<int N> __device__ __forceinline__ void ld_cg_v(const float* p, float (&v)[N]) {
    if constexpr (N == 4) {
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
    } else {
        static_assert(N == 2, "dims per lane");
        asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v[0]), "=f"(v[1]) : "l"(p));
    }
}
template<class M>
__device__ __noinline__ void attention(int l, int pos, int T) {
    constexpr int G = M::G, HD = M::HD, DPL = HD / 32, SW = HD + 4, PB = 4;
    constexpr int KVD = M::KVD, NCH = NCTA / M::NKV, NHC = M::NH * NCH;
    const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kvh = c / NCH, ch = c % NCH;
    float* qs = reinterpret_cast<float*>(Smem::uni());           // [t][G][HD] q, scaled to log2 units
    float* scr = qs + TMAX * G * HD;                             // warp states [warp][head][SW]
    const float qscale = __fmul_rn(1.44269504088896341f, C.inv_sqrt_hd);
    for (int i = tid; i < T * G * HD; i += NT) {
        const int t = i / (G * HD), r = i % (G * HD);
        qs[t * G * HD + r] = __fmul_rn(__ldcg(C.q + t * M::E + kvh * G * HD + r), qscale);
    }
    __syncthreads();
    const float* kb = C.kc + (size_t)l * C.maxseq * KVD + kvh * HD + lane * DPL;
    const float* vb = C.vc + (size_t)l * C.maxseq * KVD + kvh * HD + lane * DPL;
    for (int t = 0; t < T; t++) {
        const int Lp = pos + t + 1, p0 = ch * Lp / NCH, p1 = (ch + 1) * Lp / NCH, n = p1 - p0;
        const int a0 = p0 + n * warp / NW, a1 = p0 + n * (warp + 1) / NW;
        const int niter = ((n + NW - 1) / NW + PB - 1) / PB;      // block-uniform (longest warp share)
        const float* qt = qs + t * G * HD + lane * DPL;
        float o[G][DPL], m[G], ls[G];
        #pragma unroll
        for (int j = 0; j < G; j++) {
            m[j] = -INFINITY; ls[j] = 0.f;
            #pragma unroll
            for (int d = 0; d < DPL; d++) o[j][d] = 0.f;
        }
        for (int it = 0; it < niter; it++) {
            float kk[PB][DPL], vv[PB][DPL];
            #pragma unroll
            for (int i = 0; i < PB; i++) {      // clamped so the loads stay in range
                const int p = max(min(a0 + it * PB + i, a1 - 1), 0);
                ld_cg_v<DPL>(kb + (size_t)p * KVD, kk[i]); ld_cg_v<DPL>(vb + (size_t)p * KVD, vv[i]);
            }
            bool valid[PB];
            #pragma unroll
            for (int i = 0; i < PB; i++) valid[i] = a0 + it * PB + i < a1;
            #pragma unroll
            for (int j = 0; j < G; j++) {
                float q[DPL];
                #pragma unroll
                for (int d = 0; d < DPL; d++) q[d] = qt[j * HD + d];
                float sc[PB];
                #pragma unroll
                for (int i = 0; i < PB; i++) {
                    float dd = __fmul_rn(q[0], kk[i][0]);
                    #pragma unroll
                    for (int d = 1; d < DPL; d++) dd = __fmaf_rn(q[d], kk[i][d], dd);
                    #pragma unroll
                    for (int sft = 16; sft; sft >>= 1) dd = __fadd_rn(dd, __shfl_xor_sync(FULL, dd, sft));
                    sc[i] = dd;
                }
                // selects, not branches: invalid (clamped) positions contribute nothing
                float mx = m[j];
                #pragma unroll
                for (int i = 0; i < PB; i++) mx = valid[i] ? fmaxf(mx, sc[i]) : mx;
                const float cr = mx == -INFINITY ? 0.f : exp2f(m[j] - mx);
                float acc[DPL], l2 = __fmul_rn(ls[j], cr);
                #pragma unroll
                for (int d = 0; d < DPL; d++) acc[d] = __fmul_rn(o[j][d], cr);
                #pragma unroll
                for (int i = 0; i < PB; i++) {
                    const float pr = valid[i] ? exp2f(sc[i] - mx) : 0.f;
                    l2 = __fadd_rn(l2, pr);
                    #pragma unroll
                    for (int d = 0; d < DPL; d++) acc[d] = __fmaf_rn(pr, vv[i][d], acc[d]);
                }
                #pragma unroll
                for (int d = 0; d < DPL; d++) o[j][d] = acc[d];
                ls[j] = l2; m[j] = mx;
            }
        }
        float* sw = scr + warp * G * SW;
        #pragma unroll
        for (int j = 0; j < G; j++) {
            #pragma unroll
            for (int d = 0; d < DPL; d++) sw[j * SW + lane * DPL + d] = o[j][d];
            if (lane == 0) { sw[j * SW + HD] = m[j]; sw[j * SW + HD + 1] = ls[j]; }
        }
        __syncthreads();
        if (warp < G) {                         // warp j folds head j's 16 states in warp order
            const int j = warp;
            float ob[DPL];
            #pragma unroll
            for (int d = 0; d < DPL; d++) ob[d] = scr[j * SW + lane * DPL + d];
            float mm = scr[j * SW + HD], ll = scr[j * SW + HD + 1];
            for (int w = 1; w < NW; w++) {
                const float* s2 = scr + (w * G + j) * SW;
                const float mb = s2[HD], Mx = fmaxf(mm, mb);
                const float fa = mm == -INFINITY ? 0.f : exp2f(mm - Mx), fb = mb == -INFINITY ? 0.f : exp2f(mb - Mx);
                ll = __fmaf_rn(ll, fa, __fmul_rn(s2[HD + 1], fb));
                #pragma unroll
                for (int d = 0; d < DPL; d++) ob[d] = __fmaf_rn(ob[d], fa, __fmul_rn(s2[lane * DPL + d], fb));
                mm = Mx;
            }
            const int hp = t * NHC + (kvh * G + j) * NCH + ch;
            #pragma unroll
            for (int d = 0; d < DPL; d++) __stcg(C.ao + (size_t)hp * HD + lane * DPL + d, ob[d]);
            if (lane == 0) { __stcg(C.am + hp, mm); __stcg(C.al + hp, ll); }
        }
        __syncthreads();                        // scratch is reused by the next token
    }
}

// lm_head over row groups [lg0, lg1) for the step's T tokens: per-warp best/runner-up candidates
// (lowest row on ties) merged into smem cand[warp*TMAX + tok]
template<int CB, int XE>
__device__ __forceinline__ void lm_head_t(int lg0, int lg1, int T, int warp, int lane) {
    const int nc = C.lm.nb / CB;
    float best[2] = {-INFINITY, -INFINITY}, second[2] = {-INFINITY, -INFINITY}; int bi[2] = {INT_MAX, INT_MAX};
    for (int g = lg0 + warp; g < lg1; g += NW) {
        if (T == 1) {
            cand_push(best[0], bi[0], second[0], row_dot<CB>(C.lm, g, 0, nc, Smem::xq(), Smem::xd(), Smem::xs8(), lane), g * 32 + lane);
        } else {
            float acc[4][2] = {};
            tile_dot<CB>(C.lm, g, 0, nc, Smem::xq(), XE, Smem::xd(), Smem::xs8(), C.ts, lane, acc);
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++) cand_push(best[j], bi[j], second[j], acc[i][j], g * 32 + (lane >> 2) + 8 * i);
        }
    }
    // T = 1: every lane holds token 0; T > 1: lane holds tokens 2tig, 2tig+1 -> merge lanes of equal tig
    #pragma unroll
    for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int o = 16; o >= (T == 1 ? 1 : 4); o >>= 1)
            cand_merge(best[j], bi[j], second[j], __shfl_xor_sync(FULL, best[j], o), __shfl_xor_sync(FULL, bi[j], o), __shfl_xor_sync(FULL, second[j], o));
    if (T == 1) { if (lane == 0) Smem::cand()[warp * TMAX] = Cand{best[0], second[0], bi[0], 0}; }
    else if (lane < 4)
        for (int j = 0; j < 2; j++) Smem::cand()[warp * TMAX + 2 * lane + j] = Cand{best[j], second[j], bi[j], 0};
}
template<int XE>
__device__ __noinline__ void lm_head(int lg0, int lg1, int T, int warp, int lane) {
    if (C.lm.cb == 8) lm_head_t<8, XE>(lg0, lg1, T, warp, lane);
    else              lm_head_t<4, XE>(lg0, lg1, T, warp, lane);
}

// ---------------- speculative drafting ----------------
// Prompt-lookup drafting on the committed history h[0..len): find the latest earlier occurrence of
// the last ng tokens (ng = ngmax..1, longest first) and propose the up to kmax tokens that followed
// it. One warp; lanes test 32 candidate starts at once. Returns the draft length (warp-uniform).
__device__ int ngram_draft(const int* h, int len, int kmax, int* out, int lane) {
    for (int ng = C.ngmax; ng >= 1; ng--) {
        if (len < ng + 1) continue;
        for (int base = len - ng - 1; base >= 0; base -= 32) {
            const int j = base - lane;
            bool ok = j >= 0;
            for (int k = 0; k < ng && ok; k++) ok = h[j + k] == h[len - ng + k];
            const unsigned bal = __ballot_sync(FULL, ok);
            if (bal) {
                const int jb = base - (__ffs(bal) - 1);              // lowest lane = latest start
                const int K = min(kmax, len - (jb + ng));
                if (lane < K) out[lane] = h[jb + ng + lane];
                return K;
            }
        }
    }
    return 0;
}
// draft length that maximizes expected tokens per unit of step cost, for per-token acceptance
// probability alpha: E(K) = (1 - alpha^(K+1)) / (1 - alpha), cost(K) = 1 + slope * K
__device__ int best_draft_len(float alpha) {
    int kb = 0; float vb = 1.f, pw = alpha;
    for (int K = 1; K < TMAX; K++) {
        pw *= alpha;                                                    // alpha^(K+1)
        const float v = (1.f - pw) / (1.f - alpha) / (1.f + C.spec_slope * K);
        if (v > vb) { vb = v; kb = K; }
    }
    return kb;
}

// ---------------- the kernel ----------------
// Processes the prompt in steps of up to tchunk tokens, then generates n_gen tokens. A generation
// step runs [cur, draft_1..draft_K] (K = 0 without speculation) and commits the predictions up to
// and including the first one that disagrees with its draft — because a token's logits do not
// depend on how many tokens share the step, the output is token-for-token the plain greedy decode.
// Under TEACHER every prompt position's prediction goes to the host ring instead.
template<class M>
__global__ void __launch_bounds__(NT, 1) mega(int n_gen) {
    constexpr int E = M::E, FF = M::FF, NH = M::NH, NKV = M::NKV, HD = M::HD, KVD = M::KVD;
    constexpr int QG = M::QKV / 32, EG = E / 32, FG = FF / 32, NCH = NCTA / NKV, NHC = NH * NCH;
    constexpr int XE = M::XE, XF = M::XF, TS = M::TS, PE = XE * TMAX, PF = XF * TMAX;   // PE/PF: part offsets (floats)
    const Smem S;
    __shared__ int stok[TMAX], sdraft[TMAX], sy[TMAX], sK;
    __shared__ float smg[TMAX];
    const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // this CTA's 32-row groups of each matrix (uneven splits when the count isn't a multiple of 8)
    const int qg0 = c * QG / NCTA, nqg = (c + 1) * QG / NCTA - qg0;
    const int eg0 = c * EG / NCTA, neg = (c + 1) * EG / NCTA - eg0;
    const int fg0 = c * FG / NCTA, nfg = (c + 1) * FG / NCTA - fg0;
    const int rq = nqg * 32, ro = neg * 32, rf = nfg * 32, R0 = qg0 * 32, O0 = eg0 * 32, F0 = fg0 * 32;
    int emitted = 0, pos = 0, cur = 0, K = 0;
    bool gen = false, done = false;
    float acc_num = 0.6f, acc_den = 1.f;    // EWMA of accepted drafts / (accepted + rejections)
    unsigned long long tm = gtime();
    if (c == 0 && tid == 0) { g_clk[0] = clock64(); g_clk[1] = tm; }
    for (int i = tid; i < C.P; i += NT) S.hist()[i] = C.prompt[i];

    while (!done) {
        const int T = gen ? 1 + K : min(C.tchunk, C.P - pos);
        if (tid < T) stok[tid] = gen ? (tid == 0 ? cur : sdraft[tid - 1]) : C.prompt[pos + tid];
        __syncthreads();
        for (int l = 0; l < C.NL; l++) {
            const Layer& L = LY[l];
            // ---- P1: rmsnorm -> int8; this CTA's QKV rows (+bias, RoPE) -> q / KV cache
            if (l == 0) {                   // embedding rows (Q4_0 or Q8_0); publish this CTA's slice of h
                float* X = S.xs(XE);
                for (int i = tid; i < T * E; i += NT) {
                    const int t = i / E, e = i % E, j = e % 32;
                    const size_t blk = (size_t)stok[t] * (E / 32) + e / 32;
                    if (C.emb_q8) {
                        const uint8_t* bp = C.emb + blk * 34;
                        X[i] = __fmul_rn((float)(int8_t)bp[2 + j], __half2float(*reinterpret_cast<const __half*>(bp)));
                    } else {
                        const uint8_t* bp = C.emb + blk * 18;
                        const int qv = j < 16 ? (bp[2 + j] & 0xF) : (bp[2 + j - 16] >> 4);
                        X[i] = __fmul_rn((float)(qv - 8), __half2float(*reinterpret_cast<const __half*>(bp)));
                    }
                    if (e >= O0 && e < O0 + ro) __stcg(C.h + i, X[i]);
                }
                __syncthreads();
            } else load_tokens<M>(S.xs(XE), C.h, T);
            quant_tokens<M>(S.xs(XE), L.an, T);
            mark(tm, 0);
            gemv_part(L.qkv, qg0, nqg, PE, 0, T, XE, warp, lane);
            __syncthreads();
            mark(tm, 1);
            {
                const float* pt = S.part(XE);
                const int ks = L.qkv.ks, ss = T == 1 ? rq : TMAX * rq;   // split stride
                for (int i = tid; i < T * rq; i += NT) {
                    const int t = i / rq, r = i % rq;
                    float v = pt[t * rq + r];
                    for (int sp = 1; sp < ks; sp++) v = __fadd_rn(v, pt[sp * ss + t * rq + r]);
                    S.hb()[i] = __fadd_rn(v, __ldg(L.bqkv + R0 + r));
                }
            }
            __syncthreads();
            // NEOX RoPE on q and k heads. Their rows were permuted at load so that the rotated pair
            // (i, i + HD/2) of a head is rows (2i, 2i+1): every 32-row split keeps pairs together.
            for (int i = tid; i < T * rq / 2; i += NT) {
                const int t = i / (rq / 2), pr = i % (rq / 2), R = R0 + 2 * pr;
                if (R >= (NH + NKV) * HD) continue;                   // v rows
                const int ip = (R % HD) / 2;
                float* p = S.hb() + t * rq + 2 * pr;
                const float th = (pos + t) * powf(C.rope_base, -2.f * ip / HD), cs = cosf(th), sn = sinf(th);
                const float x0 = p[0], x1 = p[1];
                p[0] = x0 * cs - x1 * sn; p[1] = x0 * sn + x1 * cs;
            }
            __syncthreads();
            for (int i = tid; i < T * rq; i += NT) {
                const int t = i / rq, r = i % rq, R = R0 + r;
                const size_t kv = ((size_t)l * C.maxseq + pos + t) * KVD;
                if (R < NH * HD)            __stcg(C.q + t * E + R, S.hb()[i]);
                else if (R < NH * HD + KVD) __stcg(C.kc + kv + (R - NH * HD), S.hb()[i]);
                else                        __stcg(C.vc + kv + (R - NH * HD - KVD), S.hb()[i]);
            }
            grid_sync();
            mark(tm, 2);

            // ---- P2: attention partials for all T queries
            attention<M>(l, pos, T);
            mark(tm, 3);
            grid_sync();
            mark(tm, 4);

            // ---- P3: combine chunk partials -> attention output -> int8; Wo rows; residual
            if (tid < T * NHC) {                    // softmax weight of each (token, head, chunk)
                const int t = tid / NHC, hp = t * NHC + ((tid % NHC) / NCH) * NCH;
                float Mx = -INFINITY, den = 0.f;
                for (int k = 0; k < NCH; k++) Mx = fmaxf(Mx, __ldcg(C.am + hp + k));
                for (int k = 0; k < NCH; k++) {
                    const float mk = __ldcg(C.am + hp + k);
                    if (mk != -INFINITY) den = __fmaf_rn(__ldcg(C.al + hp + k), exp2f(mk - Mx), den);
                }
                const float mk = __ldcg(C.am + tid);                // partial maxima are in log2 units
                S.wts()[tid] = mk == -INFINITY ? 0.f : __fdiv_rn(exp2f(mk - Mx), den);
            }
            __syncthreads();
            {
                float* X = S.xs(XE);
                for (int i = tid; i < T * E; i += NT) {
                    const int t = i / E, idx = i % E, hp = t * NHC + (idx / HD) * NCH, e = idx % HD;
                    float acc = 0.f;
                    for (int q = 0; q < NCH; q++) acc = __fmaf_rn(__ldcg(C.ao + (size_t)(hp + q) * HD + e), S.wts()[hp + q], acc);
                    X[i] = acc;
                }
                __syncthreads();
                quant_tokens<M>(X, nullptr, T);
            }
            mark(tm, 5);
            gemv_part(L.o, eg0, neg, PE, 0, T, XE, warp, lane);
            __syncthreads();
            mark(tm, 6);
            {
                const float* pt = S.part(XE);
                const int ks = L.o.ks, ss = T == 1 ? ro : TMAX * ro;
                for (int i = tid; i < T * ro; i += NT) {
                    const int t = i / ro, r = i % ro, R = t * E + O0 + r;
                    float v = pt[t * ro + r];
                    for (int sp = 1; sp < ks; sp++) v = __fadd_rn(v, pt[sp * ss + t * ro + r]);
                    __stcg(C.h + R, __fadd_rn(__ldcg(C.h + R), v));
                }
            }
            grid_sync();
            mark(tm, 7);

            // ---- P4: rmsnorm -> int8; gate/up rows; SwiGLU -> int8 act (global)
            load_tokens<M>(S.xs(XE), C.h, T);
            quant_tokens<M>(S.xs(XE), L.fn, T);
            mark(tm, 8);
            float* gu = S.part(XE);   // T = 1: [gate splits | up splits]; T > 1: [gate][tok] | [up][tok] totals
            const int ksg = L.g.ks;   // == L.u.ks
            if (T == 1) {
                gemv_part(L.g, fg0, nfg, PE, 0, 1, XE, warp, lane);
                gemv_part(L.u, fg0, nfg, PE + ksg * rf, nfg * ksg, 1, XE, warp, lane);
            } else {
                gemv_seq(L.g, fg0, nfg, PE, 0, T, XE, warp, lane);
                gemv_seq(L.u, fg0, nfg, PE + TMAX * rf, nfg, T, XE, warp, lane);
            }
            __syncthreads();
            mark(tm, 9);
            for (int r = 0; r < (T * nfg + NW - 1) / NW; r++) {     // uniform trip count: shuffles below
                const int i = min(warp + r * NW, T * nfg - 1), t = i / nfg, b = i % nfg, j = b * 32 + lane;
                const bool own = warp + r * NW < T * nfg;
                float gv, uv;
                if (T == 1) {                   // splits summed in order, as gemv_seq does for T > 1
                    gv = gu[j]; uv = gu[ksg * rf + j];
                    for (int sp = 1; sp < ksg; sp++) { gv = __fadd_rn(gv, gu[sp * rf + j]); uv = __fadd_rn(uv, gu[(ksg + sp) * rf + j]); }
                } else { gv = gu[t * rf + j]; uv = gu[(TMAX + t) * rf + j]; }
                const float av = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.f, expf(-gv))), uv);
                float a = fabsf(av);
                #pragma unroll
                for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
                const float inv = (a > 0.f) ? 127.f / a : 0.f;
                const int q = __float2int_rn(av * inv);
                if (own) C.act_q[t * FF + F0 + j] = (int8_t)q;
                int s = q;
                #pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
                if (own && lane == 0) {
                    __stcg(C.act_d + t * TS + fg0 + b, __half2float(__float2half(a / 127.f)));
                    __stcg(C.act_s8 + t * TS + fg0 + b, 8 * s);
                }
            }
            grid_sync();
            mark(tm, 10);

            // ---- P5: act -> smem; down rows; residual
            for (int i = tid; i < T * (FF / 4); i += NT) {
                const int t = i / (FF / 4), w = i % (FF / 4);
                S.xq()[t * XF + w] = __ldcg(reinterpret_cast<const int*>(C.act_q + t * FF) + w);
            }
            for (int i = tid; i < T * TS; i += NT) { S.xd()[i] = __ldcg(C.act_d + i); S.xs8()[i] = __ldcg(C.act_s8 + i); }
            __syncthreads();
            mark(tm, 11);
            gemv_part(L.d, eg0, neg, PF, 0, T, XF, warp, lane);
            __syncthreads();
            mark(tm, 12);
            {
                const float* pt = S.part(XF);
                const int ks = L.d.ks, ss = T == 1 ? ro : TMAX * ro;
                for (int i = tid; i < T * ro; i += NT) {
                    const int t = i / ro, r = i % ro, R = t * E + O0 + r;
                    float y = 0.f;
                    for (int k = 0; k < ks; k++) y = __fadd_rn(y, pt[k * ss + t * ro + r]);
                    __stcg(C.h + R, __fadd_rn(__ldcg(C.h + R), y));
                }
            }
            grid_sync();
            mark(tm, 13);
        }
        const int first = (gen || C.teacher) ? 0 : C.P - 1 - pos;   // first token whose prediction is needed
        if (first >= T) { pos += T; continue; }    // prompt chunk before the last prompt token

        // ---- lm_head over this CTA's row groups + per-token argmax (with runner-up for the margin)
        load_tokens<M>(S.xs(XE), C.h, T);
        quant_tokens<M>(S.xs(XE), C.out_norm, T);
        mark(tm, 14);
        lm_head<XE>(c * C.glm / NCTA, (c + 1) * C.glm / NCTA, T, warp, lane);
        __syncthreads();
        if (tid < T) {
            Cand a = S.cand()[tid];
            for (int w = 1; w < NW; w++) { const Cand& b = S.cand()[w * TMAX + tid]; cand_merge(a.best, a.idx, a.second, b.best, b.idx, b.second); }
            C.cand[c * TMAX + tid] = a;
        }
        grid_sync();
        mark(tm, 15);
        if (tid < T) {                          // every CTA merges the same candidates: all agree
            float best = -INFINITY, second = -INFINITY; int bi = INT_MAX;
            for (int k = 0; k < NCTA; k++) {
                const Cand* cp = C.cand + k * TMAX + tid;
                cand_merge(best, bi, second, __ldcg(&cp->best), __ldcg(&cp->idx), __ldcg(&cp->second));
            }
            sy[tid] = bi; smg[tid] = best - second;
        }
        __syncthreads();
        auto emit = [&](int t) {                // commit prediction t of this step
            if (c == 0 && tid == 0) {           // publish to the host ring
                C.r_tok[emitted] = sy[t]; C.r_ns[emitted] = (long long)gtime(); C.r_margin[emitted] = smg[t];
                __threadfence_system();
                *(volatile int*)C.r_count = emitted + 1;
            }
            emitted++;
        };
        if (C.teacher) {
            for (int t = 0; t < T; t++) emit(t);
            pos += T;
            done = pos >= C.P;
            continue;
        }
        int a = 0;                              // accepted drafts
        if (!gen) {                             // last prompt chunk: its last prediction is the first token
            emit(first);
            cur = sy[first]; pos = C.P; gen = true;
            if (tid == 0) S.hist()[pos] = cur;
            done = (C.stop_eos && cur == C.eos) || emitted >= n_gen;
        } else {
            while (a < K && sdraft[a] == sy[a]) a++;
            for (int i = 0; i <= a && !done; i++) {
                emit(i);
                if (tid == 0) S.hist()[pos + 1 + i] = sy[i];
                done = (C.stop_eos && sy[i] == C.eos) || emitted >= n_gen;
            }
            cur = sy[a]; pos += a + 1;
            if (K > 0) {
                acc_num = 0.85f * acc_num + 0.15f * a;
                acc_den = 0.85f * acc_den + 0.15f * (a + (a < K ? 1 : 0));
            }
            if (c == 0 && tid == 0) { C.stats[0]++; C.stats[1] += K; C.stats[2] += a; }
        }
        if (done) break;
        __syncthreads();                        // history writes visible to warp 0
        if (warp == 0) {                        // draft the next step (identically in every CTA)
            int kmax = 0;
            if (C.spec) {
                const float alpha = fminf(fmaxf(acc_num / acc_den, 0.01f), 0.99f);
                kmax = min(best_draft_len(alpha), min(n_gen - emitted - 1, C.maxseq - 1 - pos));
            }
            const int k = kmax > 0 ? ngram_draft(S.hist(), pos + 1, kmax, sdraft, lane) : 0;
            if (lane == 0) sK = k;
        }
        __syncthreads();
        K = sK;
    }
    if (c == 0 && tid == 0) { g_clk[2] = clock64(); g_clk[3] = gtime(); }
}

// streams `n4` 16-byte words (the weight arena) so the DVFS governors ramp both the GPU clock
// and the memory clock before timing; without jetson_clocks the GPU idles at 306 MHz, and an
// ALU-only spin lets the EMC governor drop the memory clock instead
__global__ void warm(const uint4* p, size_t n4, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9u) *sink = acc;
}
static long gpu_mhz() {
    FILE* f = fopen("/sys/class/devfreq/17000000.gpu/cur_freq", "r");
    long hz = 0;
    if (f) { if (fscanf(f, "%ld", &hz) != 1) hz = 0; fclose(f); }
    return hz / 1000000;
}

// ---------------- host ----------------
// q4_0 requantization, same math as cuda_infer's dt_q4_0r_from (symmetric amax/7, +8, clamp)
static void quant_row_q4_0(const float* y, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; b++, y += 32, out += 18) {
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(y[j]));
        const float d = amax / 7.f, inv = (d > 0.f) ? 1.f / d : 0.f;
        const __half hd = __float2half(d); memcpy(out, &hd, 2);
        for (int j = 0; j < 16; j++) {
            int q0 = (int)lrintf(y[j] * inv) + 8;      q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            int q1 = (int)lrintf(y[j + 16] * inv) + 8; q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            out[2 + j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

// nvmap does not wait for page-cache reclaim: a big cudaMalloc fails while MemAvailable says it fits.
// Faulting the same amount of host memory first makes the kernel reclaim, then the retry succeeds.
static void* dev_alloc(size_t bytes) {
    for (int attempt = 0;; attempt++) {
        void* p;
        if (cudaMalloc(&p, bytes) == cudaSuccess) return p;
        cudaGetLastError();
        if (attempt == 3) { fprintf(stderr, "mega: cudaMalloc(%zu MB) failed — free RAM (tmpfs models?)\n", bytes >> 20); exit(1); }
        void* h = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (h != MAP_FAILED) munmap(h, bytes);
    }
}
struct Arena {                                       // one device allocation for every weight
    uint8_t* base = nullptr; size_t used = 0, cap = 0;
    template<class T> T* put(const void* src, size_t bytes) {
        size_t off = (used + 255) & ~(size_t)255;
        if (off + bytes > cap) { fprintf(stderr, "mega: arena overflow\n"); exit(1); }
        CK(cudaMemcpy(base + off, src, bytes, cudaMemcpyHostToDevice));
        used = off + bytes;
        return reinterpret_cast<T*>(base + off);
    }
};
static void release_pages(const Tensor& t) {           // drop the GGUF mmap pages once uploaded
    long ps = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)t.data, start = a & ~(uintptr_t)(ps - 1);
    madvise((void*)start, (a - start) + t.nbytes(), MADV_DONTNEED);
}
// row-major Q4_0 (18-byte blocks) -> 32-row-group layout with chunks of cb blocks (see PM)
static PM pack_q4(Arena& A, const uint8_t* src, int rows, int nb, int ks) {
    const int ng = rows / 32, cb = nb % 8 == 0 ? 8 : 4, nc = nb / cb;
    std::vector<uint4> q((size_t)rows * nb);
    std::vector<uint16_t> sc((size_t)rows * nb);
    for (int g = 0; g < ng; g++)
        for (int b = 0; b < nb; b++)
            for (int r = 0; r < 32; r++)
                memcpy(&q[((size_t)g * nb + b) * 32 + r], src + ((size_t)(g * 32 + r) * nb + b) * 18 + 2, 16);
    for (int g = 0; g < ng; g++)
        for (int cc = 0; cc < nc; cc++)
            for (int r = 0; r < 32; r++)
                for (int k = 0; k < cb; k++)
                    memcpy(&sc[(((size_t)g * nc + cc) * 32 + r) * cb + k], src + ((size_t)(g * 32 + r) * nb + cc * cb + k) * 18, 2);
    PM m;
    m.q = A.put<uint4>(q.data(), q.size() * 16);
    m.s = A.put<__half>(sc.data(), sc.size() * 2);
    m.nb = nb; m.cb = cb; m.ks = ks;
    return m;
}
// a weight tensor as row-major Q4_0: as stored, or requantized (any other type) into buf
static const uint8_t* q4_rows(const Tensor& t, std::vector<uint8_t>& buf) {
    if (t.type == GT_Q4_0) return t.data;
    const int n = (int)t.ne[0], rows = (int)t.rows(), nb = n / 32;
    buf.resize((size_t)rows * nb * 18);
    std::vector<float> row(n);
    for (int r = 0; r < rows; r++) { GGUF::dequant_row(t, r, row.data()); quant_row_q4_0(row.data(), n, buf.data() + (size_t)r * nb * 18); }
    return buf.data();
}
// split-K count for a matrix whose CTAs own up to ngmax row groups of nc chunks each (joint:
// matrices issued back to back, e.g. gate+up): fewest task rounds x chunks per task, +0.5 per task
static int choose_ks(int ngmax, int nc, int joint) {
    int best = 1; double bc = 1e30;
    for (int ks = 1; ks <= nc; ks++) {
        const double cost = std::ceil((double)joint * ngmax * ks / NW) * (std::ceil((double)nc / ks) + 0.5);
        if (cost < bc - 1e-9) { bc = cost; best = ks; }
    }
    return best;
}

template<class M>
static int run(GGUF& g, int n_gen, const std::vector<int>& ids, bool raw_ids) {
    constexpr int E = M::E, FF = M::FF, NH = M::NH, NKV = M::NKV, HD = M::HD, KVD = M::KVD, QKV = M::QKV, G = M::G;
    constexpr int NHC = NH * (NCTA / NKV), TS = M::TS;
    Cfg cfg{};
    cfg.NL = (int)g.meta_num("qwen2.block_count", 0);
    cfg.rope_base = (float)g.meta_num("qwen2.rope.freq_base", 1e6);
    cfg.eps = (float)g.meta_num("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    cfg.inv_sqrt_hd = 1.f / sqrtf((float)HD);
    cfg.eos = (int)g.meta_num("tokenizer.ggml.eos_token_id", 151645);
    cfg.P = (int)ids.size();
    cfg.teacher = getenv("TEACHER") ? atoi(getenv("TEACHER")) : 0;
    cfg.stop_eos = !raw_ids && !cfg.teacher;
    cfg.prof = getenv("PROFILE") ? 1 : 0;
    cfg.tchunk = getenv("TCHUNK") ? atoi(getenv("TCHUNK")) : TMAX;
    cfg.spec = getenv("SPEC") ? atoi(getenv("SPEC")) : 1;          // n-gram speculative decoding (lossless)
    cfg.ngmax = getenv("NGMAX") ? atoi(getenv("NGMAX")) : 3;
    cfg.spec_slope = getenv("SPEC_SLOPE") ? (float)atof(getenv("SPEC_SLOPE")) : 0.074f;   // measured: T=8 step = 1.52x
    cfg.maxseq = cfg.P + n_gen + TMAX;      // + TMAX: a step writes K/V ahead for its drafts
    cfg.ts = TS;
    const int V = (int)g.tokens().size();
    REQUIRE(cfg.NL >= 1 && cfg.NL <= 64 && V % 32 == 0 && cfg.tchunk >= 1 && cfg.tchunk <= TMAX && cfg.P >= 1);
    REQUIRE(NCTA % NKV == 0 && TMAX * NHC <= NT && (E / 32) % 4 == 0 && (FF / 32) % 4 == 0);
    cfg.glm = V / 32;

    // split-K per matrix kind, and the dynamic smem layout: the union holds the phase's int8
    // activations (T tokens at a padded word stride) and its split partials
    auto gmax = [](int ngroups) { return (ngroups + NCTA - 1) / NCTA; };
    auto ncof = [](int nb) { return nb / (nb % 8 == 0 ? 8 : 4); };
    const int nbE = E / 32, nbF = FF / 32, QG = QKV / 32, EG = E / 32, FG = FF / 32;
    const int ks_qkv = choose_ks(gmax(QG), ncof(nbE), 1), ks_o = choose_ks(gmax(EG), ncof(nbE), 1);
    const int ks_gu = choose_ks(gmax(FG), ncof(nbE), 2), ks_d = choose_ks(gmax(EG), ncof(nbF), 1);
    const int rqmax = gmax(QG) * 32, romax = gmax(EG) * 32, rfmax = gmax(FG) * 32;
    size_t uni = 0;
    auto need = [&](size_t b) { if (b > uni) uni = b; };
    const size_t xe = (size_t)M::XE * TMAX * 4, xf = (size_t)M::XF * TMAX * 4;
    need(xe + (size_t)TMAX * E * 4);                                     // staged token vectors
    need(xe + (size_t)ks_qkv * TMAX * rqmax * 4);                        // P1
    need(xe + (size_t)ks_o * TMAX * romax * 4);                          // P3
    need(xe + (size_t)std::max(2 * ks_gu, 2 * TMAX) * rfmax * 4);        // P4
    need(xf + (size_t)ks_d * TMAX * romax * 4);                          // P5
    need(((size_t)TMAX * G * HD + (size_t)NW * G * (HD + 4)) * 4);       // attention: q + warp states
    auto al16 = [](size_t b) { return (b + 15) & ~(size_t)15; };
    cfg.off_xd = (int)al16(uni);
    cfg.off_xs8 = cfg.off_xd + TMAX * TS * 4;
    cfg.off_hb = cfg.off_xs8 + TMAX * TS * 4;
    cfg.off_red = (int)al16(cfg.off_hb + (size_t)TMAX * rqmax * 4);
    cfg.off_wts = (int)al16(cfg.off_red + TMAX * NW * 4);
    cfg.off_cand = (int)al16(cfg.off_wts + (size_t)TMAX * NHC * 4);
    cfg.off_scale = (int)al16(cfg.off_cand + (size_t)NW * TMAX * sizeof(Cand));
    cfg.off_hist = (int)al16(cfg.off_scale + TMAX * 4);
    const size_t smem_bytes = cfg.off_hist + (size_t)cfg.maxseq * 4;

    int dev = 0, sms = 0, coop = 0, smax = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    CK(cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    REQUIRE(sms == NCTA && coop);
    if (smem_bytes > (size_t)smax) {
        fprintf(stderr, "mega: needs %zu B of shared memory per CTA (max %d) — shorten prompt + n_gen\n", smem_bytes, smax);
        return 1;
    }
    CK(cudaFuncSetAttribute(mega<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    // ---- weights -> one arena (packed q4 groups), norms/biases alongside
    const Tensor& te = g.get("token_embd.weight");
    const Tensor& tl = g.has("output.weight") ? g.get("output.weight") : te;   // tied embeddings
    size_t bytes = 0;
    auto q4b = [](size_t rows, int nb) { return rows * nb * 18 + 1024; };
    bytes += cfg.NL * (q4b(QKV, nbE) + q4b(E, nbE) + 2 * q4b(FF, nbE) + q4b(E, nbF) + (QKV + 2 * E) * 4 + 3 * 256);
    bytes += q4b(V, nbE) + (size_t)V * nbE * (te.type == GT_Q8_0 ? 34 : 18) + E * 4 + 4096;
    Arena A; A.cap = bytes; A.base = (uint8_t*)dev_alloc(bytes);

    std::vector<Layer> ly(cfg.NL);
    std::vector<uint8_t> cat((size_t)QKV * nbE * 18), tmp;
    std::vector<float> bias(QKV);
    // q and k rows (and biases) go to pair-adjacent order: row 2i of a head <- row i, row 2i+1 <-
    // row i + HD/2, so the RoPE pair of every rotated element sits in the same 32-row group
    auto perm = [](int r) { const int h = r / HD, i = r % HD; return h * HD + (i < HD / 2 ? 2 * i : 2 * (i - HD / 2) + 1); };
    for (int l = 0; l < cfg.NL; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const size_t rb = (size_t)nbE * 18;
        const uint8_t* wq = q4_rows(g.get(p + "attn_q.weight"), tmp);
        for (int r = 0; r < NH * HD; r++) memcpy(cat.data() + perm(r) * rb, wq + r * rb, rb);
        const uint8_t* wk = q4_rows(g.get(p + "attn_k.weight"), tmp);
        for (int r = 0; r < KVD; r++) memcpy(cat.data() + (NH * HD + perm(r)) * rb, wk + r * rb, rb);
        const uint8_t* wv = q4_rows(g.get(p + "attn_v.weight"), tmp);
        memcpy(cat.data() + (size_t)(NH * HD + KVD) * rb, wv, (size_t)KVD * rb);
        ly[l].qkv = pack_q4(A, cat.data(), QKV, nbE, ks_qkv);
        const float* bq = (const float*)g.get(p + "attn_q.bias").data;
        const float* bk = (const float*)g.get(p + "attn_k.bias").data;
        for (int r = 0; r < NH * HD; r++) bias[perm(r)] = bq[r];
        for (int r = 0; r < KVD; r++) bias[NH * HD + perm(r)] = bk[r];
        memcpy(bias.data() + NH * HD + KVD, g.get(p + "attn_v.bias").data, KVD * 4);
        ly[l].bqkv = A.put<float>(bias.data(), QKV * 4);
        ly[l].o = pack_q4(A, q4_rows(g.get(p + "attn_output.weight"), tmp), E, nbE, ks_o);
        ly[l].g = pack_q4(A, q4_rows(g.get(p + "ffn_gate.weight"), tmp), FF, nbE, ks_gu);
        ly[l].u = pack_q4(A, q4_rows(g.get(p + "ffn_up.weight"), tmp), FF, nbE, ks_gu);
        ly[l].d = pack_q4(A, q4_rows(g.get(p + "ffn_down.weight"), tmp), E, nbF, ks_d);
        ly[l].an = A.put<float>(g.get(p + "attn_norm.weight").data, E * 4);
        ly[l].fn = A.put<float>(g.get(p + "ffn_norm.weight").data, E * 4);
        for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                              "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) release_pages(g.get(p + n));
    }
    cfg.lm = pack_q4(A, q4_rows(tl, tmp), V, nbE, 1);       // lm_head (Q6_K/Q8_0 heads requantized to q4_0)
    cfg.emb_q8 = te.type == GT_Q8_0;
    cfg.emb = A.put<uint8_t>(te.type == GT_Q8_0 ? te.data : q4_rows(te, tmp), (size_t)V * nbE * (cfg.emb_q8 ? 34 : 18));
    release_pages(tl); release_pages(te);
    cfg.out_norm = A.put<float>(g.get("output_norm.weight").data, E * 4);

    // ---- state (per-token buffers sized for TMAX tokens)
    auto dalloc = [](size_t b) { void* p = dev_alloc(b); CK(cudaMemset(p, 0, b)); return p; };
    cfg.h = (float*)dalloc((size_t)TMAX * E * 4);
    cfg.q = (float*)dalloc((size_t)TMAX * E * 4);
    cfg.am = (float*)dalloc((size_t)TMAX * NHC * 4); cfg.al = (float*)dalloc((size_t)TMAX * NHC * 4);
    cfg.ao = (float*)dalloc((size_t)TMAX * NHC * HD * 4);
    cfg.act_q = (int8_t*)dalloc((size_t)TMAX * FF);
    cfg.act_d = (float*)dalloc((size_t)TMAX * TS * 4); cfg.act_s8 = (int*)dalloc((size_t)TMAX * TS * 4);
    cfg.kc = (float*)dalloc((size_t)cfg.NL * cfg.maxseq * KVD * 4);
    cfg.vc = (float*)dalloc((size_t)cfg.NL * cfg.maxseq * KVD * 4);
    cfg.cand = (Cand*)dalloc((size_t)NCTA * TMAX * sizeof(Cand));
    cfg.stats = (long long*)dalloc(4 * 8);
    int* dprompt = (int*)dalloc(ids.size() * 4);
    CK(cudaMemcpy(dprompt, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    cfg.prompt = dprompt;
    int *h_count, *h_tok; long long* h_ns; float* h_margin;           // mapped ring
    CK(cudaHostAlloc(&h_count, 4, cudaHostAllocMapped)); *h_count = 0;
    const int nring = cfg.P + n_gen;
    CK(cudaHostAlloc(&h_tok, nring * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc(&h_ns, nring * 8, cudaHostAllocMapped));
    CK(cudaHostAlloc(&h_margin, nring * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&cfg.r_count, h_count, 0));
    CK(cudaHostGetDevicePointer((void**)&cfg.r_tok, h_tok, 0));
    CK(cudaHostGetDevicePointer((void**)&cfg.r_ns, h_ns, 0));
    CK(cudaHostGetDevicePointer((void**)&cfg.r_margin, h_margin, 0));
    CK(cudaMemcpyToSymbol(C, &cfg, sizeof cfg));
    CK(cudaMemcpyToSymbol(LY, ly.data(), ly.size() * sizeof(Layer)));
    fprintf(stderr, "minfer-mega: qwen2 L=%d d=%d ff=%d H=%d/%d hd=%d vocab=%d | weights %.1f MB in one arena | %d CTAs x %d thr, smem %zu B, "
            "split-K qkv/o/gate-up/down %d/%d/%d/%d, prompt chunk %d\n",
            cfg.NL, E, FF, NH, NKV, HD, V, A.used / 1e6, NCTA, NT, smem_bytes, ks_qkv, ks_o, ks_gu, ks_d, cfg.tchunk);

    // ---- run: one cooperative launch for the prompt + every generated token
    const std::vector<std::string>& vocab = g.tokens();
    int cp2byte[324]; build_byte_decoder(cp2byte);
    int ngen_arg = n_gen;
    void* args[] = {&ngen_arg};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    {   // WARMUP=<ms> (default 1500): ramp the GPU clock before the timed run
        const int ms = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 1500;
        const long mhz0 = gpu_mhz();
        unsigned* sink = (unsigned*)dalloc(4);
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < ms) {
            warm<<<NCTA * 6, 256, 0, s>>>(reinterpret_cast<const uint4*>(A.base), A.used / 16, sink);   // ~10 ms per pass
            CK(cudaStreamSynchronize(s));
        }
        fprintf(stderr, "minfer-mega: GPU clock %ld MHz -> %ld MHz after %d ms warm-up\n", mhz0, gpu_mhz(), ms);
    }
    auto t0 = std::chrono::steady_clock::now();
    CK(cudaLaunchCooperativeKernel((void*)mega<M>, NCTA, NT, args, smem_bytes, s));
    int seen = 0; double ttft_ms = -1;
    for (;;) {
        const int cnt = __atomic_load_n(h_count, __ATOMIC_ACQUIRE);
        for (; seen < cnt; seen++) {
            if (ttft_ms < 0) ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const int id = ((volatile int*)h_tok)[seen];
            if (cfg.teacher) continue;
            if (raw_ids) printf("%d ", id);
            else if (id != cfg.eos && id < V) { std::string o; detok_piece(vocab[id], cp2byte, o); fputs(o.c_str(), stdout); }
            fflush(stdout);
        }
        if (cudaStreamQuery(s) != cudaErrorNotReady && seen == __atomic_load_n(h_count, __ATOMIC_ACQUIRE)) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    CK(cudaStreamSynchronize(s));
    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("\n");
    const int n = seen;
    if (n > 1 && !cfg.teacher) {
        const double ms = (h_ns[n - 1] - h_ns[0]) / 1e6;
        float minm = 1e30f; int mi = 0;
        for (int i = 0; i < n; i++) if (h_margin[i] < minm) { minm = h_margin[i]; mi = i; }
        fprintf(stderr, "minfer-mega decode: %d tok in %.1f ms = %.2f tok/s | first token after %.1f ms (prompt %d) | min logit margin %.4f at #%d\n",
                n - 1, ms, (n - 1) * 1e3 / ms, ttft_ms, cfg.P, minm, mi);
    }
    if (cfg.teacher) {          // tok[i] = prediction for position i+1; compare with the reference
        int agree = 0, total = 0;
        for (int i = cfg.teacher - 1; i + 1 < cfg.P; i++, total++) {
            if (h_tok[i] == ids[i + 1]) { agree++; continue; }
            fprintf(stderr, "  pos %d: ref %d, mega %d (mega margin %.4f)\n", i + 1, ids[i + 1], h_tok[i], h_margin[i]);
        }
        unsigned long long hsh = 1469598103934665603ull;     // FNV-1a over (token, margin bits): exactness check
        for (int i = 0; i < n; i++) {
            unsigned mb; memcpy(&mb, &h_margin[i], 4);
            for (unsigned v : {(unsigned)h_tok[i], mb}) for (int k = 0; k < 4; k++) { hsh ^= (v >> (8 * k)) & 0xff; hsh *= 1099511628211ull; }
        }
        fprintf(stderr, "teacher-forced top-1 agreement: %d/%d | %d positions in %.1f ms (%.2f ms per %d-token step) | logit hash %016llx\n",
                agree, total, cfg.P, total_ms, total_ms * cfg.tchunk / cfg.P, cfg.tchunk, hsh);
    }
    if (cfg.prof) {
        unsigned long long pr[32]; CK(cudaMemcpyFromSymbol(pr, g_prof, sizeof pr));
        const char* ph[6] = {"P1 QKV", "P2 attention", "P3 Wo", "P4 gate/up", "P5 down", "lm_head"};
        const int first[7] = {0, 3, 5, 8, 11, 14, 16};
        double tot = 0;
        const double steps = (double)(cfg.teacher ? cfg.P : cfg.P - 1 + n);
        for (int k = 0; k < 6; k++) {
            const double div = k < 5 ? steps : (double)n;
            fprintf(stderr, "  %-13s", ph[k]);
            double sum = 0;
            for (int i = first[k]; i < first[k + 1]; i++) { const double us = pr[i] / 1e3 / div; sum += us; fprintf(stderr, " %8.1f", us); }
            fprintf(stderr, "  = %8.1f us/token\n", sum);
            tot += sum * (k < 5 ? 1.0 : (double)n / steps);
        }
        fprintf(stderr, "  (columns: prologue, gemv, epilogue+barrier; P2: compute, barrier)  total %.1f us/position\n", tot);
        fprintf(stderr, "  attention parts (us/position): q stage %.1f, loads+softmax %.1f, merge %.1f\n",
                pr[16] / 1e3 / steps, pr[17] / 1e3 / steps, pr[18] / 1e3 / steps);
    }
    {
        unsigned long long k[4]; CK(cudaMemcpyFromSymbol(k, g_clk, sizeof k));
        fprintf(stderr, "minfer-mega: effective SM clock during the run %.0f MHz (GPU devfreq now %ld MHz)\n",
                (double)(k[2] - k[0]) * 1e3 / (double)(k[3] - k[1]), gpu_mhz());
    }
    if (!cfg.teacher) {
        long long st[3]; CK(cudaMemcpy(st, cfg.stats, sizeof st, cudaMemcpyDeviceToHost));
        if (st[0] > 0)
            fprintf(stderr, "minfer-mega: %lld generation steps, %.2f tokens/step | drafted %lld, accepted %lld (%.0f%%)%s\n",
                    st[0], (double)(n - 1) / st[0], st[1], st[2], st[1] ? 100.0 * st[2] / st[1] : 0.0, cfg.spec ? "" : " [SPEC=0]");
    }
    if (getenv("SEGMENTS") && n > 1) {     // tok/s over consecutive windows: shows the slope with depth
        const int w = atoi(getenv("SEGMENTS"));
        for (int i = 0; i + w < n; i += w)
            fprintf(stderr, "  tokens %4d-%4d (pos %4d): %.1f tok/s\n", i, i + w, cfg.P + i, w * 1e9 / (h_ns[i + w] - h_ns[i]));
    }
    if (getenv("MARGINS")) for (int i = 0; i < n; i++) fprintf(stderr, "%d:%d:%.4f%c", i, h_tok[i], h_margin[i], i + 1 < n ? ' ' : '\n');
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gen> <id...>\n", argv[0]); return 1; }
    GGUF g(argv[1]);
    const int n_gen = atoi(argv[2]);
    std::vector<int> ids; for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));
    const bool raw_ids = getenv("RAWIDS") && atoi(getenv("RAWIDS"));
    const int E = (int)g.meta_num("qwen2.embedding_length", 0), FF = (int)g.meta_num("qwen2.feed_forward_length", 0);
    const int NH = (int)g.meta_num("qwen2.attention.head_count", 0), NKV = (int)g.meta_num("qwen2.attention.head_count_kv", 0);
    auto is = [&](auto m) { using S = decltype(m); return E == S::E && FF == S::FF && NH == S::NH && NKV == S::NKV; };
    if (is(Qwen05{})) return run<Qwen05>(g, n_gen, ids, raw_ids);
    if (is(Qwen15{})) return run<Qwen15>(g, n_gen, ids, raw_ids);
    if (is(Qwen3{}))  return run<Qwen3>(g, n_gen, ids, raw_ids);
    fprintf(stderr, "mega: no kernel instantiation for d=%d ff=%d heads=%d/%d (add a Shape)\n", E, FF, NH, NKV);
    return 1;
}
