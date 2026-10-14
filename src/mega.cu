// minfer M15/M16 — the whole decode loop as ONE persistent kernel.
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
// M16: a step can carry T <= 8 tokens (prompt chunks now; speculative verify next). T = 1
// keeps the dp4a path; T > 1 runs the same packed weights through tensor cores
// (mma.m16n8k32 s8), so one weight pass serves all T tokens. Both paths form identical
// per-block integer sums and accumulate them through the same explicitly rounded
// expression in the same order, so a token's logits do not depend on T.
// Numerics follow cuda_infer (q4_0 x int8 activations, same quantization, fp32 KV cache);
// only float summation order differs.
// Result (Orin Nano Super, unpinned clocks): ~97 tok/s vs cuda_infer's ~70 at short context,
// 89 vs 50 tok/s averaged over 2048 tokens; teacher-forced top-1 agreement with cuda_infer
// 99.3% (cuda_infer vs itself with split-KV attention: 99.8%), every miss a near-tie.
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
constexpr int MAXE = 1536, MAXFF = 8960, TS = MAXFF / 32;   // TS: per-token stride of block scales
constexpr unsigned FULL = 0xffffffffu;

// q4_0 matrix packed in 32-row groups: lane r of a warp owns row g*32+r.
//   q[(g*nb + b)*32 + r]     = the 16 nibble bytes of row g*32+r, block b
//   s[(g*nb/8 + c)*32 + r]   = that row's 8 fp16 scales for blocks 8c..8c+7
struct PM { const uint4* q; const uint4* s; int nb; };
struct Layer { PM qkv, o, g, u, d; const float *an, *fn, *bqkv; };
struct Cand { float best, second; int idx, pad; };
struct Cfg {
    int E, FF, NH, NKV, HD, NL, maxseq, P, eos, stop_eos, teacher, prof, tchunk, spec, ngmax;
    float spec_slope;                       // cost of each extra token in a step, relative to a 1-token step
    int gqkv, go, gff, glm;                 // row groups per CTA (lm: total groups)
    int xe, xf;                             // smem word stride per token: E- / FF-wide activations
    int off_xd, off_xs8, off_hb, off_red, off_wts, off_cand, off_scale, off_hist;   // dynamic smem layout (bytes)
    float eps, rope_base, inv_sqrt_hd;
    PM lm; const float* out_norm; const uint8_t* emb;   // emb: raw Q4_0 rows (18-byte blocks)
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
// T = 1: lane's row of group g over block chunks [c0, c1) (8 blocks each), token 0 of xq/xd/xs8
__device__ __forceinline__ float row_dot(const PM& M, int g, int c0, int c1, const int* xq, const float* xd, const int* xs8, int lane) {
    const int nc = M.nb / 8;
    const uint4* qp = M.q + ((size_t)g * M.nb + c0 * 8) * 32 + lane;
    const uint4* sp = M.s + ((size_t)g * nc + c0) * 32 + lane;
    float acc = 0.f;
    for (int c = c0; c < c1; c++, qp += 256, sp += 32) {
        uint4 w[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) w[k] = ldw(qp + k * 32);
        const uint4 sc = ldw(sp);
        const __half2* sh = reinterpret_cast<const __half2*>(&sc);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int b = c * 8 + k;
            const float dw = (k & 1) ? __high2float(sh[k >> 1]) : __low2float(sh[k >> 1]);
            acc = blk_acc(acc, dw, xd[b], dot_q4(w[k], xq + b * 8), xs8[b]);
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
__device__ __forceinline__ void tile_dot(const PM& M, int g, int c0, int c1, const int* xq, int xs,
                                         const float* xd, const int* xs8, int lane, float (&acc)[4][2]) {
    const int gid = lane >> 2, tig = lane & 3, nc = M.nb / 8;
    const unsigned* qw = reinterpret_cast<const unsigned*>(M.q + (size_t)g * M.nb * 32) + gid * 4 + tig;
    const uint4* sp = M.s + (size_t)g * nc * 32 + gid;
    const int* xb = xq + gid * xs + tig;
    const unsigned MK = 0x0F0F0F0Fu;
    for (int c = c0; c < c1; c++) {
        uint4 sc[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) sc[i] = ldw(sp + (size_t)c * 32 + 8 * i);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int b = c * 8 + k;
            unsigned w[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) w[i] = ldw32(qw + ((size_t)b * 32 + 8 * i) * 4);
            const unsigned b0 = xb[b * 8], b1 = xb[b * 8 + 4];
            int d0[4], d1[4];
            mma_s8(d0, w[0] & MK, w[1] & MK, (w[0] >> 4) & MK, (w[1] >> 4) & MK, b0, b1);   // rows gid, gid+8
            mma_s8(d1, w[2] & MK, w[3] & MK, (w[2] >> 4) & MK, (w[3] >> 4) & MK, b0, b1);   // rows gid+16, gid+24
            const float dx0 = xd[(2 * tig) * TS + b], dx1 = xd[(2 * tig + 1) * TS + b];
            const int s0 = xs8[(2 * tig) * TS + b], s1 = xs8[(2 * tig + 1) * TS + b];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const __half2 h2 = reinterpret_cast<const __half2*>(&sc[i])[k >> 1];
                const float dw = (k & 1) ? __high2float(h2) : __low2float(h2);
                const int* d = i < 2 ? d0 : d1;
                const int e = (i & 1) * 2;          // d[e], d[e+1] = (row gid + 8i, tokens 2tig, 2tig+1)
                acc[i][0] = blk_acc(acc[i][0], dw, dx0, d[e], s0);
                acc[i][1] = blk_acc(acc[i][1], dw, dx1, d[e + 1], s1);
            }
        }
    }
}
// split-K GEMV over row groups [g0, g0+ng): task = (group, split), task t runs on warp
// (t0 + t) % NW so consecutive calls keep the warps balanced.
//   T = 1: part[split*ng*32 + row]                 T > 1: part[(split*TMAX + tok)*ng*32 + row]
__device__ __noinline__ void gemv_part(const PM& M, int g0, int ng, int ks, float* part, int t0, int T, const Smem& S, int xs, int warp, int lane) {
    const int per = M.nb / 8 / ks, ntask = ng * ks, gid = lane >> 2, tig = lane & 3;
    for (int t = ((warp - t0) % NW + NW) % NW; t < ntask; t += NW) {
        const int gl = t / ks, sp = t % ks;
        if (T == 1) {
            part[sp * ng * 32 + gl * 32 + lane] = row_dot(M, g0 + gl, sp * per, sp * per + per, S.xq(), S.xd(), S.xs8(), lane);
            continue;
        }
        float acc[4][2] = {};
        tile_dot(M, g0 + gl, sp * per, sp * per + per, S.xq(), xs, S.xd(), S.xs8(), lane, acc);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                if (2 * tig + j < T) part[(sp * TMAX + 2 * tig + j) * ng * 32 + gl * 32 + gid + 8 * i] = acc[i][j];
    }
}
// T > 1 for the gate/up phase: a warp runs every split of its group and adds them in split
// order — the same (P0 + P1) the T = 1 path forms from part[] — so only the totals need smem.
// out[tok*ng*32 + row]
__device__ __noinline__ void gemv_seq(const PM& M, int g0, int ng, int ks, float* out, int t0, int T, const Smem& S, int xs, int warp, int lane) {
    const int per = M.nb / 8 / ks, gid = lane >> 2, tig = lane & 3;
    for (int t = ((warp - t0) % NW + NW) % NW; t < ng; t += NW) {
        float tot[4][2];
        for (int sp = 0; sp < ks; sp++) {
            float acc[4][2] = {};
            tile_dot(M, g0 + t, sp * per, sp * per + per, S.xq(), xs, S.xd(), S.xs8(), lane, acc);
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
// symmetric int8 per 32-block, exactly cuda_infer's quant_x_q8 (lane = element)
__device__ __forceinline__ void quant32(float v, int b, int8_t* xq, float* xd, int* xs8, int lane) {
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
    const float inv = (a > 0.f) ? 127.f / a : 0.f;
    const int q = __float2int_rn(v * inv);
    xq[b * 32 + lane] = (int8_t)q;
    int s = q;
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (lane == 0) { xd[b] = __half2float(__float2half(a / 127.f)); xs8[b] = 8 * s; }
}
// token vectors X[t][E] (t < T, staged in smem) -> optionally rmsnorm(x_t) * w -> int8 blocks of
// each token in smem. Every loop that holds a shuffle has a block-uniform trip count so the warp
// stays provably converged; each token gets exactly the arithmetic of a lone token.
__device__ __forceinline__ void quant_tokens(const Smem& S, const float* X, const float* w, int T, int xs) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, n = C.E, nb = n / 32;
    if (w) {
        for (int t = 0; t < T; t++) {
            float ss = 0.f;
            for (int i = tid; i < n; i += NT) ss = __fmaf_rn(X[t * n + i], X[t * n + i], ss);
            #pragma unroll
            for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(FULL, ss, o));
            if (lane == 0) S.red()[t * NW + warp] = ss;
        }
        __syncthreads();
        if (tid < T) {
            float s = 0.f;
            #pragma unroll
            for (int v = 0; v < NW; v++) s = __fadd_rn(s, S.red()[tid * NW + v]);
            S.scale()[tid] = rsqrtf(__fadd_rn(__fdiv_rn(s, (float)n), C.eps));
        }
        __syncthreads();
    }
    for (int r = 0; r < T * nb / NW; r++) {            // (token, block) pairs; nb % NW == 0 (REQUIREd)
        const int i = warp + r * NW, t = i / nb, b = i % nb, e = b * 32 + lane;
        const float v = w ? __fmul_rn(__fmul_rn(X[t * n + e], S.scale()[t]), __ldg(w + e)) : X[t * n + e];
        quant32(v, b, reinterpret_cast<int8_t*>(S.xq() + t * xs), S.xd() + t * TS, S.xs8() + t * TS, lane);
    }
    __syncthreads();
}
__device__ __forceinline__ void load_tokens(float* X, const float* src, int T) {
    for (int i = threadIdx.x; i < T * C.E; i += NT) X[i] = __ldcg(src + i);
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
// for the G q heads of the kv head at once (one K/V read serves every head; lane = 4 dims), PB
// positions per batch: scores are kept in log2 units (q pre-scaled by log2(e)/sqrt(hd)) and the
// running state is rescaled once per batch. All 16 warp states then go to smem and warp j folds
// head j's states in warp order into partial [t][head][chunk] (m in log2 units). The split and the
// fold order depend only on the token's position, so a token's numbers do not depend on how many
// tokens share the step. Loops that hold shuffles run a block-uniform trip count and branch only
// around shuffle-free code, so the warp stays provably converged (otherwise every SHFL becomes a
// call that serializes the heads' reductions).
__device__ __noinline__ void attention(const Smem& S, int l, int pos, int T) {
    constexpr int G = 6, PB = 4;            // q heads per kv head (compile-time, REQUIREd); positions per batch
    const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int HD = C.HD, KVD = C.NKV * HD, NCH = NCTA / C.NKV, NHC = C.NH * NCH;
    const int kvh = c / NCH, ch = c % NCH;
    float* qs = reinterpret_cast<float*>(S.uni());               // [t][G][128] q, scaled to log2 units
    float* scr = qs + TMAX * G * 128;                            // warp states [warp][head][132]
    const float qscale = __fmul_rn(1.44269504088896341f, C.inv_sqrt_hd);
    for (int i = tid; i < T * G * HD; i += NT) {
        const int t = i / (G * HD), r = i % (G * HD);
        qs[t * G * 128 + r] = __fmul_rn(__ldcg(C.q + t * C.E + kvh * G * HD + r), qscale);
    }
    __syncthreads();
    const float4* kb = reinterpret_cast<const float4*>(C.kc + (size_t)l * C.maxseq * KVD + kvh * HD) + lane;
    const float4* vb = reinterpret_cast<const float4*>(C.vc + (size_t)l * C.maxseq * KVD + kvh * HD) + lane;
    const size_t st = KVD / 4;
    for (int t = 0; t < T; t++) {
        const int Lp = pos + t + 1, p0 = ch * Lp / NCH, p1 = (ch + 1) * Lp / NCH, n = p1 - p0;
        const int a0 = p0 + n * warp / NW, a1 = p0 + n * (warp + 1) / NW;
        const int niter = ((n + NW - 1) / NW + PB - 1) / PB;      // block-uniform (longest warp share)
        const float4* qt = reinterpret_cast<const float4*>(qs + t * G * 128) + lane;
        float4 o[G]; float m[G], ls[G];
        #pragma unroll
        for (int j = 0; j < G; j++) { o[j] = make_float4(0.f, 0.f, 0.f, 0.f); m[j] = -INFINITY; ls[j] = 0.f; }
        for (int it = 0; it < niter; it++) {
            float4 kk[PB], vv[PB];
            #pragma unroll
            for (int i = 0; i < PB; i++) {      // clamped so the loads stay in range
                const int p = max(min(a0 + it * PB + i, a1 - 1), 0);
                kk[i] = ld_cg4(kb + p * st); vv[i] = ld_cg4(vb + p * st);
            }
            bool valid[PB];
            #pragma unroll
            for (int i = 0; i < PB; i++) valid[i] = a0 + it * PB + i < a1;
            #pragma unroll
            for (int j = 0; j < G; j++) {
                const float4 q = qt[j * 32];
                float sc[PB];
                #pragma unroll
                for (int i = 0; i < PB; i++) {
                    float d = __fmul_rn(q.x, kk[i].x);
                    d = __fmaf_rn(q.y, kk[i].y, d); d = __fmaf_rn(q.z, kk[i].z, d); d = __fmaf_rn(q.w, kk[i].w, d);
                    #pragma unroll
                    for (int sft = 16; sft; sft >>= 1) d = __fadd_rn(d, __shfl_xor_sync(FULL, d, sft));
                    sc[i] = d;
                }
                // selects, not branches: invalid (clamped) positions contribute nothing
                float mx = m[j];
                #pragma unroll
                for (int i = 0; i < PB; i++) mx = valid[i] ? fmaxf(mx, sc[i]) : mx;
                const float cr = mx == -INFINITY ? 0.f : exp2f(m[j] - mx);
                float4 acc = make_float4(__fmul_rn(o[j].x, cr), __fmul_rn(o[j].y, cr), __fmul_rn(o[j].z, cr), __fmul_rn(o[j].w, cr));
                float l2 = __fmul_rn(ls[j], cr);
                #pragma unroll
                for (int i = 0; i < PB; i++) {
                    const float pr = valid[i] ? exp2f(sc[i] - mx) : 0.f;
                    l2 = __fadd_rn(l2, pr);
                    acc.x = __fmaf_rn(pr, vv[i].x, acc.x); acc.y = __fmaf_rn(pr, vv[i].y, acc.y);
                    acc.z = __fmaf_rn(pr, vv[i].z, acc.z); acc.w = __fmaf_rn(pr, vv[i].w, acc.w);
                }
                o[j] = acc; ls[j] = l2; m[j] = mx;
            }
        }
        float* sw = scr + warp * G * 132;
        #pragma unroll
        for (int j = 0; j < G; j++) {
            *reinterpret_cast<float4*>(sw + j * 132 + lane * 4) = o[j];
            if (lane == 0) { sw[j * 132 + 128] = m[j]; sw[j * 132 + 129] = ls[j]; }
        }
        __syncthreads();
        if (warp < G) {                         // warp j folds head j's 16 states in warp order
            const int j = warp;
            float4 ob = *reinterpret_cast<const float4*>(scr + j * 132 + lane * 4);
            float mm = scr[j * 132 + 128], ll = scr[j * 132 + 129];
            for (int w = 1; w < NW; w++) {
                const float* s2 = scr + (w * G + j) * 132;
                const float mb = s2[128], M = fmaxf(mm, mb);
                const float fa = mm == -INFINITY ? 0.f : exp2f(mm - M), fb = mb == -INFINITY ? 0.f : exp2f(mb - M);
                const float4 v2 = *reinterpret_cast<const float4*>(s2 + lane * 4);
                ll = __fmaf_rn(ll, fa, __fmul_rn(s2[129], fb));
                ob.x = __fmaf_rn(ob.x, fa, __fmul_rn(v2.x, fb)); ob.y = __fmaf_rn(ob.y, fa, __fmul_rn(v2.y, fb));
                ob.z = __fmaf_rn(ob.z, fa, __fmul_rn(v2.z, fb)); ob.w = __fmaf_rn(ob.w, fa, __fmul_rn(v2.w, fb));
                mm = M;
            }
            const int hp = t * NHC + (kvh * G + j) * NCH + ch;
            __stcg(reinterpret_cast<float4*>(C.ao + (size_t)hp * HD) + lane, ob);
            if (lane == 0) { __stcg(C.am + hp, mm); __stcg(C.al + hp, ll); }
        }
        __syncthreads();                        // scratch is reused by the next token
    }
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
__global__ void __launch_bounds__(NT, 1) mega(int n_gen) {
    const Smem S;
    __shared__ int stok[TMAX], sdraft[TMAX], sy[TMAX], sK;
    __shared__ float smg[TMAX];
    const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int E = C.E, HD = C.HD, NH = C.NH, KVD = C.NKV * HD, NCH = NCTA / C.NKV, NHC = NH * NCH;
    const int rq = C.gqkv * 32, ro = C.go * 32, rf = C.gff * 32, XE = C.xe, XF = C.xf;
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
            if (l == 0) {                   // embedding rows (Q4_0); publish this CTA's slice of h
                float* X = S.xs(XE);
                for (int i = tid; i < T * E; i += NT) {
                    const int t = i / E, e = i % E, j = e % 32;
                    const int tok = stok[t];
                    const uint8_t* bp = C.emb + ((size_t)tok * (E / 32) + e / 32) * 18;
                    const int qv = j < 16 ? (bp[2 + j] & 0xF) : (bp[2 + j - 16] >> 4);
                    X[i] = __fmul_rn((float)(qv - 8), __half2float(*reinterpret_cast<const __half*>(bp)));
                    if (e >= c * ro && e < (c + 1) * ro) __stcg(C.h + i, X[i]);
                }
                __syncthreads();
            } else load_tokens(S.xs(XE), C.h, T);
            quant_tokens(S, S.xs(XE), L.an, T, XE);
            mark(tm, 0);
            gemv_part(L.qkv, c * C.gqkv, C.gqkv, 2, S.part(XE), 0, T, S, XE, warp, lane);
            __syncthreads();
            mark(tm, 1);
            const int R0 = c * rq;
            {
                const float* pt = S.part(XE);
                const int s1 = T == 1 ? rq : TMAX * rq;          // stride to split 1
                for (int i = tid; i < T * rq; i += NT) {
                    const int t = i / rq, r = i % rq;
                    S.hb()[i] = __fadd_rn(__fadd_rn(pt[t * rq + r], pt[s1 + t * rq + r]), __ldg(L.bqkv + R0 + r));
                }
            }
            __syncthreads();
            for (int i = tid; i < T * rq / 2; i += NT) {    // NEOX RoPE on q and k heads (whole heads per CTA)
                const int t = i / (rq / 2), hh = (i % (rq / 2)) / (HD / 2), j = i % (HD / 2);
                if (R0 + hh * HD >= (NH + C.NKV) * HD) continue;   // v head
                float* p = S.hb() + t * rq + hh * HD;
                const float th = (pos + t) * powf(C.rope_base, -2.f * j / HD), cs = cosf(th), sn = sinf(th);
                const float x0 = p[j], x1 = p[j + HD / 2];
                p[j] = x0 * cs - x1 * sn; p[j + HD / 2] = x0 * sn + x1 * cs;
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
            attention(S, l, pos, T);
            mark(tm, 3);
            grid_sync();
            mark(tm, 4);

            // ---- P3: combine chunk partials -> attention output -> int8; Wo rows; residual
            if (tid < T * NHC) {                    // softmax weight of each (token, head, chunk)
                const int t = tid / NHC, hp = t * NHC + ((tid % NHC) / NCH) * NCH;
                float M = -INFINITY, den = 0.f;
                for (int k = 0; k < NCH; k++) M = fmaxf(M, __ldcg(C.am + hp + k));
                for (int k = 0; k < NCH; k++) {
                    const float mk = __ldcg(C.am + hp + k);
                    if (mk != -INFINITY) den = __fmaf_rn(__ldcg(C.al + hp + k), exp2f(mk - M), den);
                }
                const float mk = __ldcg(C.am + tid);                // partial maxima are in log2 units
                S.wts()[tid] = mk == -INFINITY ? 0.f : __fdiv_rn(exp2f(mk - M), den);
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
                quant_tokens(S, X, nullptr, T, XE);
            }
            mark(tm, 5);
            gemv_part(L.o, c * C.go, C.go, 2, S.part(XE), 0, T, S, XE, warp, lane);
            __syncthreads();
            mark(tm, 6);
            {
                const float* pt = S.part(XE);
                const int s1 = T == 1 ? ro : TMAX * ro;
                for (int i = tid; i < T * ro; i += NT) {
                    const int t = i / ro, r = i % ro, R = t * E + c * ro + r;
                    __stcg(C.h + R, __fadd_rn(__ldcg(C.h + R), __fadd_rn(pt[t * ro + r], pt[s1 + t * ro + r])));
                }
            }
            grid_sync();
            mark(tm, 7);

            // ---- P4: rmsnorm -> int8; gate/up rows; SwiGLU -> int8 act (global)
            load_tokens(S.xs(XE), C.h, T);
            quant_tokens(S, S.xs(XE), L.fn, T, XE);
            mark(tm, 8);
            float* gu = S.part(XE);   // T = 1: [gate s0 | gate s1 | up s0 | up s1]; T > 1: [gate][tok] | [up][tok] totals
            if (T == 1) {
                gemv_part(L.g, c * C.gff, C.gff, 2, gu, 0, 1, S, XE, warp, lane);
                gemv_part(L.u, c * C.gff, C.gff, 2, gu + 2 * rf, 2 * C.gff, 1, S, XE, warp, lane);
            } else {
                gemv_seq(L.g, c * C.gff, C.gff, 2, gu, 0, T, S, XE, warp, lane);
                gemv_seq(L.u, c * C.gff, C.gff, 2, gu + TMAX * rf, C.gff, T, S, XE, warp, lane);
            }
            __syncthreads();
            mark(tm, 9);
            for (int r = 0; r < (T * (rf / 32) + NW - 1) / NW; r++) {     // uniform trip count: shuffles below
                const int i = min(warp + r * NW, T * (rf / 32) - 1), t = i / (rf / 32), b = i % (rf / 32), j = b * 32 + lane;
                const bool own = warp + r * NW < T * (rf / 32);
                const float gv = T == 1 ? __fadd_rn(gu[j], gu[rf + j]) : gu[t * rf + j];
                const float uv = T == 1 ? __fadd_rn(gu[2 * rf + j], gu[3 * rf + j]) : gu[(TMAX + t) * rf + j];
                const float av = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.f, expf(-gv))), uv);
                float a = fabsf(av);
                #pragma unroll
                for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
                const float inv = (a > 0.f) ? 127.f / a : 0.f;
                const int q = __float2int_rn(av * inv);
                if (own) C.act_q[t * C.FF + c * rf + j] = (int8_t)q;
                int s = q;
                #pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
                if (own && lane == 0) {
                    __stcg(C.act_d + t * TS + c * C.gff + b, __half2float(__float2half(a / 127.f)));
                    __stcg(C.act_s8 + t * TS + c * C.gff + b, 8 * s);
                }
            }
            grid_sync();
            mark(tm, 10);

            // ---- P5: act -> smem; down rows; residual
            for (int i = tid; i < T * (C.FF / 4); i += NT) {
                const int t = i / (C.FF / 4), w = i % (C.FF / 4);
                S.xq()[t * XF + w] = __ldcg(reinterpret_cast<const int*>(C.act_q + t * C.FF) + w);
            }
            for (int i = tid; i < T * TS; i += NT) { S.xd()[i] = __ldcg(C.act_d + i); S.xs8()[i] = __ldcg(C.act_s8 + i); }
            __syncthreads();
            mark(tm, 11);
            gemv_part(L.d, c * C.go, C.go, 5, S.part(XF), 0, T, S, XF, warp, lane);
            __syncthreads();
            mark(tm, 12);
            {
                const float* pt = S.part(XF);
                const int ss = T == 1 ? ro : TMAX * ro;          // split stride
                for (int i = tid; i < T * ro; i += NT) {
                    const int t = i / ro, r = i % ro, R = t * E + c * ro + r;
                    float y = 0.f;
                    for (int k = 0; k < 5; k++) y = __fadd_rn(y, pt[k * ss + t * ro + r]);
                    __stcg(C.h + R, __fadd_rn(__ldcg(C.h + R), y));
                }
            }
            grid_sync();
            mark(tm, 13);
        }
        const int first = (gen || C.teacher) ? 0 : C.P - 1 - pos;   // first token whose prediction is needed
        if (first >= T) { pos += T; continue; }    // prompt chunk before the last prompt token

        // ---- lm_head over this CTA's row groups + per-token argmax (with runner-up for the margin)
        load_tokens(S.xs(XE), C.h, T);
        quant_tokens(S, S.xs(XE), C.out_norm, T, XE);
        mark(tm, 14);
        {
            const int lg0 = c * C.glm / NCTA, lg1 = (c + 1) * C.glm / NCTA, nc = C.lm.nb / 8;
            float best[2] = {-INFINITY, -INFINITY}, second[2] = {-INFINITY, -INFINITY}; int bi[2] = {INT_MAX, INT_MAX};
            for (int g = lg0 + warp; g < lg1; g += NW) {
                if (T == 1) {
                    cand_push(best[0], bi[0], second[0], row_dot(C.lm, g, 0, nc, S.xq(), S.xd(), S.xs8(), lane), g * 32 + lane);
                } else {
                    float acc[4][2] = {};
                    tile_dot(C.lm, g, 0, nc, S.xq(), XE, S.xd(), S.xs8(), lane, acc);
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
            if (T == 1) { if (lane == 0) S.cand()[warp * TMAX] = Cand{best[0], second[0], bi[0], 0}; }
            else if (lane < 4)
                for (int j = 0; j < 2; j++) S.cand()[warp * TMAX + 2 * lane + j] = Cand{best[j], second[j], bi[j], 0};
            __syncthreads();
            if (tid < T) {
                Cand a = S.cand()[tid];
                for (int w = 1; w < NW; w++) { const Cand& b = S.cand()[w * TMAX + tid]; cand_merge(a.best, a.idx, a.second, b.best, b.idx, b.second); }
                C.cand[c * TMAX + tid] = a;
            }
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
// row-major Q4_0 (18-byte blocks) -> 32-row-group layout (see PM)
static PM pack_q4(Arena& A, const uint8_t* src, int rows, int nb) {
    const int ng = rows / 32;
    std::vector<uint4> q((size_t)rows * nb), s((size_t)rows * nb / 8);
    for (int g = 0; g < ng; g++)
        for (int b = 0; b < nb; b++)
            for (int r = 0; r < 32; r++)
                memcpy(&q[((size_t)g * nb + b) * 32 + r], src + ((size_t)(g * 32 + r) * nb + b) * 18 + 2, 16);
    for (int g = 0; g < ng; g++)
        for (int cc = 0; cc < nb / 8; cc++)
            for (int r = 0; r < 32; r++) {
                uint16_t sc[8];
                for (int k = 0; k < 8; k++) memcpy(&sc[k], src + ((size_t)(g * 32 + r) * nb + cc * 8 + k) * 18, 2);
                memcpy(&s[((size_t)g * (nb / 8) + cc) * 32 + r], sc, 16);
            }
    PM m;
    m.q = A.put<uint4>(q.data(), q.size() * 16);
    m.s = A.put<uint4>(s.data(), s.size() * 16);
    m.nb = nb;
    return m;
}
static const uint8_t* q4_rows(const Tensor& t) {
    if (t.type != GT_Q4_0) { fprintf(stderr, "mega: expected Q4_0 weights\n"); exit(1); }
    return t.data;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gen> <id...>\n", argv[0]); return 1; }
    GGUF g(argv[1]);
    const int n_gen = atoi(argv[2]);
    std::vector<int> ids; for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));
    const bool raw_ids = getenv("RAWIDS") && atoi(getenv("RAWIDS"));

    Cfg cfg{};
    cfg.NL = (int)g.meta_num("qwen2.block_count", 0);
    cfg.E = (int)g.meta_num("qwen2.embedding_length", 0);
    cfg.FF = (int)g.meta_num("qwen2.feed_forward_length", 0);
    cfg.NH = (int)g.meta_num("qwen2.attention.head_count", 0);
    cfg.NKV = (int)g.meta_num("qwen2.attention.head_count_kv", 0);
    cfg.HD = cfg.E / cfg.NH;
    cfg.rope_base = (float)g.meta_num("qwen2.rope.freq_base", 1e6);
    cfg.eps = (float)g.meta_num("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    cfg.inv_sqrt_hd = 1.f / sqrtf((float)cfg.HD);
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
    const int V = (int)g.tokens().size(), KVD = cfg.NKV * cfg.HD, QKV = (cfg.NH + 2 * cfg.NKV) * cfg.HD;
    REQUIRE(cfg.E <= MAXE && cfg.FF <= MAXFF && cfg.HD == 128 && cfg.NL <= 64);
    REQUIRE(cfg.E % 256 == 0 && cfg.FF % 256 == 0 && V % 32 == 0 && cfg.tchunk >= 1 && cfg.tchunk <= TMAX && cfg.P >= 1);
    REQUIRE(QKV % (32 * NCTA) == 0 && (QKV / NCTA) % cfg.HD == 0);     // whole heads per CTA
    REQUIRE(cfg.E % (32 * NCTA) == 0 && cfg.FF % (32 * NCTA) == 0);
    REQUIRE(NCTA % cfg.NKV == 0 && cfg.NH == 6 * cfg.NKV && (cfg.E / 256) % 2 == 0 && (cfg.FF / 256) % 5 == 0);
    cfg.gqkv = QKV / 32 / NCTA; cfg.go = cfg.E / 32 / NCTA; cfg.gff = cfg.FF / 32 / NCTA; cfg.glm = V / 32;
    REQUIRE(cfg.E % NT == 0 && (cfg.E / 32) % NW == 0 && cfg.E == cfg.NH * cfg.HD && TMAX * cfg.NH * (NCTA / cfg.NKV) <= NT);

    // dynamic smem layout: the union holds the phase's int8 activations (T tokens at a padded
    // word stride: +4 keeps the mma B-fragment loads bank-conflict free) and its split partials
    const int rq = cfg.gqkv * 32, ro = cfg.go * 32, rf = cfg.gff * 32;
    cfg.xe = cfg.E / 4 + 4; cfg.xf = cfg.FF / 4 + 4;
    size_t uni = 0;
    auto need = [&](size_t b) { if (b > uni) uni = b; };
    need((size_t)cfg.xe * TMAX * 4 + (size_t)TMAX * cfg.E * 4);          // staged token vectors
    need((size_t)cfg.xe * TMAX * 4 + (size_t)2 * TMAX * rq * 4);        // P1
    need((size_t)cfg.xe * TMAX * 4 + (size_t)2 * TMAX * ro * 4);        // P3
    need((size_t)cfg.xe * TMAX * 4 + (size_t)std::max(2 * TMAX, 4) * rf * 4);   // P4
    need((size_t)cfg.xf * TMAX * 4 + (size_t)5 * TMAX * ro * 4);        // P5
    need(((size_t)TMAX * 6 * 128 + NW * 6 * 132) * 4);                  // attention: q + warp states
    auto al16 = [](size_t b) { return (b + 15) & ~(size_t)15; };
    const size_t hbn = std::max<size_t>(cfg.E, (size_t)TMAX * rq);
    cfg.off_xd = (int)al16(uni);
    cfg.off_xs8 = cfg.off_xd + TMAX * TS * 4;
    cfg.off_hb = cfg.off_xs8 + TMAX * TS * 4;
    cfg.off_red = (int)al16(cfg.off_hb + hbn * 4);
    cfg.off_wts = (int)al16(cfg.off_red + TMAX * NW * 4);
    cfg.off_cand = (int)al16(cfg.off_wts + (size_t)TMAX * cfg.NH * (NCTA / cfg.NKV) * 4);
    cfg.off_scale = (int)al16(cfg.off_cand + (size_t)NW * TMAX * sizeof(Cand));
    cfg.off_hist = (int)al16(cfg.off_scale + TMAX * 4);
    const size_t smem_bytes = cfg.off_hist + (size_t)cfg.maxseq * 4;

    int dev = 0, sms = 0, coop = 0, smax = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    CK(cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    REQUIRE(sms == NCTA && coop && smem_bytes <= (size_t)smax);
    CK(cudaFuncSetAttribute(mega, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    // ---- weights -> one arena (packed q4 groups), norms/biases alongside
    const int nbE = cfg.E / 32, nbF = cfg.FF / 32;
    size_t bytes = 0;
    auto q4b = [](size_t rows, int nb) { return rows * nb * 18 + 1024; };
    bytes += cfg.NL * (q4b(QKV, nbE) + q4b(cfg.E, nbE) + 2 * q4b(cfg.FF, nbE) + q4b(cfg.E, nbF) + (QKV + 2 * cfg.E) * 4 + 3 * 256);
    bytes += q4b(V, nbE) + (size_t)V * nbE * 18 + cfg.E * 4 + 4096;
    Arena A; A.cap = bytes; CK(cudaMalloc(&A.base, bytes));

    std::vector<Layer> ly(cfg.NL);
    std::vector<uint8_t> cat((size_t)QKV * nbE * 18);
    std::vector<float> bias(QKV);
    for (int l = 0; l < cfg.NL; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const Tensor &tq = g.get(p + "attn_q.weight"), &tk = g.get(p + "attn_k.weight"), &tv = g.get(p + "attn_v.weight");
        memcpy(cat.data(), q4_rows(tq), tq.nbytes());
        memcpy(cat.data() + tq.nbytes(), q4_rows(tk), tk.nbytes());
        memcpy(cat.data() + tq.nbytes() + tk.nbytes(), q4_rows(tv), tv.nbytes());
        ly[l].qkv = pack_q4(A, cat.data(), QKV, nbE);
        memcpy(bias.data(), g.get(p + "attn_q.bias").data, cfg.E * 4);
        memcpy(bias.data() + cfg.E, g.get(p + "attn_k.bias").data, KVD * 4);
        memcpy(bias.data() + cfg.E + KVD, g.get(p + "attn_v.bias").data, KVD * 4);
        ly[l].bqkv = A.put<float>(bias.data(), QKV * 4);
        ly[l].o = pack_q4(A, q4_rows(g.get(p + "attn_output.weight")), cfg.E, nbE);
        ly[l].g = pack_q4(A, q4_rows(g.get(p + "ffn_gate.weight")), cfg.FF, nbE);
        ly[l].u = pack_q4(A, q4_rows(g.get(p + "ffn_up.weight")), cfg.FF, nbE);
        ly[l].d = pack_q4(A, q4_rows(g.get(p + "ffn_down.weight")), cfg.E, nbF);
        ly[l].an = A.put<float>(g.get(p + "attn_norm.weight").data, cfg.E * 4);
        ly[l].fn = A.put<float>(g.get(p + "ffn_norm.weight").data, cfg.E * 4);
        for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                              "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) release_pages(g.get(p + n));
    }
    {   // lm_head: Q6_K -> q4_0 rows (as cuda_infer LMHEAD=4) -> packed
        const Tensor& t = g.get("output.weight");
        std::vector<uint8_t> rows((size_t)V * nbE * 18);
        std::vector<float> row(cfg.E);
        for (int r = 0; r < V; r++) {
            GGUF::dequant_row(t, r, row.data());
            quant_row_q4_0(row.data(), cfg.E, rows.data() + (size_t)r * nbE * 18);
        }
        cfg.lm = pack_q4(A, rows.data(), V, nbE);
        release_pages(t);
    }
    const Tensor& te = g.get("token_embd.weight");
    REQUIRE(te.type == GT_Q4_0);
    cfg.emb = A.put<uint8_t>(te.data, te.nbytes());
    release_pages(te);
    cfg.out_norm = A.put<float>(g.get("output_norm.weight").data, cfg.E * 4);

    // ---- state (per-token buffers sized for TMAX tokens)
    auto dalloc = [](size_t b) { void* p; CK(cudaMalloc(&p, b)); CK(cudaMemset(p, 0, b)); return p; };
    const int NHC = cfg.NH * (NCTA / cfg.NKV);
    cfg.h = (float*)dalloc((size_t)TMAX * cfg.E * 4);
    cfg.q = (float*)dalloc((size_t)TMAX * cfg.E * 4);
    cfg.am = (float*)dalloc((size_t)TMAX * NHC * 4); cfg.al = (float*)dalloc((size_t)TMAX * NHC * 4);
    cfg.ao = (float*)dalloc((size_t)TMAX * NHC * cfg.HD * 4);
    cfg.act_q = (int8_t*)dalloc((size_t)TMAX * cfg.FF);
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
    fprintf(stderr, "minfer-mega: qwen2 L=%d d=%d ff=%d H=%d/%d vocab=%d | weights %.1f MB in one arena | %d CTAs x %d thr, smem %zu B, prompt chunk %d\n",
            cfg.NL, cfg.E, cfg.FF, cfg.NH, cfg.NKV, V, A.used / 1e6, NCTA, NT, smem_bytes, cfg.tchunk);

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
    CK(cudaLaunchCooperativeKernel((void*)mega, NCTA, NT, args, smem_bytes, s));
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
