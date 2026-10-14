// bench/roofline.cu — Phase 0: what this Orin's memory system can actually do.
//
// The old llm-optimization/bandwidth_test read 6.88 MB per launch x 500 launches with
// ~1.6 loads per thread, so launch + ramp + tail dominated and it reported 75.2 GB/s.
// Here every bandwidth number is ONE launch over a buffer far bigger than the 2 MB L2,
// with several independent 16-byte loads in flight per thread. Also measured: DRAM
// latency (sizes the megakernel's prefetch ring by Little's law), the cost of an
// 8-CTA grid barrier, and the per-node gap of a CUDA graph (what minfer pays ~567x/token).
//
//   nvcc -O3 -arch=sm_87 -o build/roofline bench/roofline.cu && ./build/roofline
// Run after `sudo jetson_clocks` (the EMC clock unpins after sleep), ideally headless.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

// ---------------- load flavours ----------------
// 0: ld.global   1: ld.global.nc.L1::no_allocate   2: 1 + L2::256B prefetch hint
template<int MODE> __device__ __forceinline__ uint4 ld16(const uint4* p) {
    uint4 v;
    if (MODE == 0)
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else if (MODE == 1)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned fold(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

// whole grid sweeps the buffer together, U loads in flight per thread
template<int MODE, int U>
__global__ void k_gridstride(const uint4* __restrict__ p, size_t n4, unsigned* sink) {
    const size_t nth = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i + (U - 1) * nth < n4; i += U * nth) {
        uint4 v[U];
        #pragma unroll
        for (int u = 0; u < U; u++) v[u] = ld16<MODE>(p + i + u * nth);
        #pragma unroll
        for (int u = 0; u < U; u++) acc ^= fold(v[u]);
    }
    for (; i < n4; i += nth) acc ^= fold(ld16<MODE>(p + i));
    if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;   // never true; keeps the loads live
}

// each block streams its own contiguous 1/gridDim slice (the megakernel's per-SM stream)
template<int MODE, int U>
__global__ void k_chunked(const uint4* __restrict__ p, size_t n4, unsigned* sink) {
    const size_t per = n4 / gridDim.x, end = (blockIdx.x + 1) * per, nb = blockDim.x;
    size_t i = blockIdx.x * per + threadIdx.x;
    unsigned acc = 0;
    for (; i + (U - 1) * nb < end; i += U * nb) {
        uint4 v[U];
        #pragma unroll
        for (int u = 0; u < U; u++) v[u] = ld16<MODE>(p + i + u * nb);
        #pragma unroll
        for (int u = 0; u < U; u++) acc ^= fold(v[u]);
    }
    for (; i < end; i += nb) acc ^= fold(ld16<MODE>(p + i));
    if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;
}

// ---------------- cp.async ring (Ampere LDGSTS) ----------------
// CPM 0: cp.async.cg   1: + L2::256B prefetch   2: + L2 evict_first cache policy
template<int CPM> __device__ __forceinline__ void cp16(void* smem, const void* g, uint64_t pol) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    if (CPM == 0)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory");
    else if (CPM == 1)
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory");
    else
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" :: "r"(s), "l"(g), "l"(pol) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// one block per SM streams its contiguous slice through an NST-stage smem ring
// (CUTLASS-multistage style: refill the slot freed last iteration, then consume).
template<int CPM, int NST>
__global__ void k_cpasync(const uint4* __restrict__ p, size_t n4, int stage4, unsigned* sink) {
    extern __shared__ uint4 ring[];
    uint64_t pol = 0;
    if (CPM == 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const size_t per = n4 / gridDim.x, nst = per / stage4;
    const uint4* src = p + blockIdx.x * per;
    const int tid = threadIdx.x, nt = blockDim.x;
    unsigned acc = 0;
    for (int s = 0; s < NST - 1; s++) {
        if ((size_t)s < nst)
            for (int j = tid; j < stage4; j += nt) cp16<CPM>(&ring[s * stage4 + j], src + (size_t)s * stage4 + j, pol);
        cp_commit();
    }
    for (size_t s = 0; s < nst; s++) {
        cp_wait<NST - 2>();
        __syncthreads();                         // stage s landed for all; slot of s-1 is free
        size_t ns = s + NST - 1;
        if (ns < nst) {
            int slot = (int)(ns % NST);
            for (int j = tid; j < stage4; j += nt) cp16<CPM>(&ring[slot * stage4 + j], src + ns * stage4 + j, pol);
        }
        cp_commit();
        int slot = (int)(s % NST);
        for (int j = tid; j < stage4; j += nt) acc ^= fold(ring[slot * stage4 + j]);
    }
    if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;
}

// ---------------- latency, barrier, graph gap ----------------
__global__ void k_chase(const unsigned* __restrict__ next, int hops, unsigned start, unsigned* out) {
    unsigned j = start;
    for (int h = 0; h < hops; h++) j = __ldcg(next + j);   // L2-cached load; L1 can't help a random walk
    *out = j;
}

__device__ unsigned g_bar_count, g_bar_gen;
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v; asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
// sense-reversing grid barrier: read gen, arrive, last arriver resets + bumps gen
__device__ __forceinline__ void grid_barrier() {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned gen = ld_acquire(&g_bar_gen);
        if (atomicAdd(&g_bar_count, 1u) == gridDim.x - 1) {
            atomicExch(&g_bar_count, 0u);
            st_release(&g_bar_gen, gen + 1);
        } else {
            while (ld_acquire(&g_bar_gen) == gen) {}
        }
    }
    __syncthreads();
}
__global__ void k_barrier(int iters, unsigned* sink) {
    for (int i = 0; i < iters; i++) grid_barrier();
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = g_bar_gen;
}
__global__ void k_empty() {}
__global__ void k_tiny(float* x) { x[blockIdx.x * blockDim.x + threadIdx.x] += 1.f; }

// ---------------- harness ----------------
static cudaEvent_t ev0, ev1;
template<class F> static float time_ms(F launch, int reps = 5) {      // best of reps (after 1 warmup)
    launch(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(ev0)); launch(); CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
        float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1)); best = std::min(best, ms);
    }
    CK(cudaGetLastError());
    return best;
}
static void row(const char* name, size_t bytes, float ms) {
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("  %-46s %7.2f ms  %6.1f GB/s  (%3.0f%% of 102.4)\n", name, ms, gbs, 100.0 * gbs / 102.4);
}

int main(int argc, char** argv) {
    size_t MB = argc > 1 ? (size_t)atoi(argv[1]) : 768;
    size_t bytes = MB << 20, n4 = bytes / 16;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

    int dev = 0, sms, coop, tmo, l2, smem_optin, maxthr, gclk = 0;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    CK(cudaDeviceGetAttribute(&tmo, cudaDevAttrKernelExecTimeout, dev));
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    CK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CK(cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    cudaDeviceGetAttribute(&gclk, cudaDevAttrClockRate, dev);
    printf("%s sm_%d%d  SMs=%d  L2=%d KB  smem/block(optin)=%d KB  threads/SM=%d  clock(attr)=%d MHz\n",
           prop.name, prop.major, prop.minor, sms, l2 >> 10, smem_optin >> 10, maxthr, gclk / 1000);
    printf("cooperativeLaunch=%d  kernelExecTimeout=%d   buffer=%zu MB\n\n", coop, tmo, MB);

    uint4* buf; CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 0x5A, bytes));
    unsigned* sink; CK(cudaMalloc(&sink, 4096 * 4));

    int bs = 256, gs_blocks = sms * (maxthr / bs);         // full occupancy grid
    // unpinned clocks: the EMC governor sits at a low step until it sees sustained load,
    // which would bias whichever test runs first. Stream for ~1.5 s before measuring.
    for (float t = 0; t < 1500.f; ) t += time_ms([&]{ k_chunked<2, 8><<<sms * 6, 256>>>(buf, n4, sink); }, 1);
    printf("[1] grid-stride read, %d blocks x %d thr, U = loads in flight/thread\n", gs_blocks, bs);
    #define GS(M, U, name) row(name, bytes, time_ms([&]{ k_gridstride<M, U><<<gs_blocks, bs>>>(buf, n4, sink); }))
    GS(0, 1, "ld.global            U=1");
    GS(0, 4, "ld.global            U=4");
    GS(1, 4, "ld.nc.L1::no_alloc   U=4");
    GS(2, 4, "ld.nc + L2::256B     U=4");
    GS(2, 8, "ld.nc + L2::256B     U=8");

    printf("\n[2] per-SM contiguous slices (megakernel access pattern)\n");
    #define CH(M, U, nb, nt, name) row(name, bytes, time_ms([&]{ k_chunked<M, U><<<nb, nt>>>(buf, n4, sink); }))
    CH(2, 4, sms, 1024, "1 blk/SM x 1024 thr, nc+256B U=4");
    CH(2, 8, sms, 1024, "1 blk/SM x 1024 thr, nc+256B U=8");
    CH(2, 8, sms * 6, 256, "6 blk/SM x 256 thr,  nc+256B U=8");
    printf("  SM-count sweep (1 blk/SM x 1024 thr, nc+256B U=8):\n");
    for (int k : {1, 2, 4, 6}) {
        char nm[64]; snprintf(nm, sizeof nm, "  only %d SM%s", k, k > 1 ? "s" : "");
        size_t nk4 = n4 / sms * k;                          // same bytes per SM as the full run
        float ms = time_ms([&]{ k_chunked<2, 8><<<k, 1024>>>(buf, nk4, sink); });
        row(nm, nk4 * 16, ms);
    }

    printf("\n[3] cp.async.cg -> smem ring, 1 blk/SM x 512 thr (stage x stages = ring)\n");
    auto cpa = [&](auto kern, int stage_kb, int nst, const char* name) {
        int smem = stage_kb * 1024 * nst, st4 = stage_kb * 1024 / 16;
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        char nm[80]; snprintf(nm, sizeof nm, "%s %2d KB x %d = %3d KB", name, stage_kb, nst, stage_kb * nst);
        row(nm, n4 / sms / st4 * st4 * sms * 16, time_ms([&]{ kern<<<sms, 512, smem>>>(buf, n4, st4, sink); }));
    };
    cpa(k_cpasync<0, 4>,  8, 4, "cg              ");
    cpa(k_cpasync<0, 8>,  8, 8, "cg              ");
    cpa(k_cpasync<0, 8>, 16, 8, "cg              ");
    cpa(k_cpasync<1, 8>, 16, 8, "cg+L2::256B     ");
    cpa(k_cpasync<2, 8>, 16, 8, "cg+evict_first  ");
    cpa(k_cpasync<1, 4>, 32, 4, "cg+L2::256B     ");

    printf("\n[4] size sweep (grid-stride, nc+256B U=8) — TLB / L2 effects\n");
    for (size_t mb : {4, 16, 64, 256}) {
        if (mb > MB) break;
        char nm[64]; snprintf(nm, sizeof nm, "%4zu MB", mb);
        size_t m4 = (mb << 20) / 16;
        row(nm, mb << 20, time_ms([&]{ k_gridstride<2, 8><<<gs_blocks, bs>>>(buf, m4, sink); }, 20));
    }

    printf("\n[5] dependent-load latency (single-thread random pointer chase, 256 B nodes)\n");
    for (size_t mb : {1, 32, 256}) {
        size_t nodes = (mb << 20) / 256, stride = 256 / 4;
        std::vector<unsigned> h(nodes * stride, 0), perm(nodes);
        for (size_t i = 0; i < nodes; i++) perm[i] = (unsigned)i;
        std::mt19937 rng(42);
        for (size_t i = nodes - 1; i > 0; i--) std::swap(perm[i], perm[rng() % i]);   // Sattolo: one cycle
        for (size_t i = 0; i < nodes; i++) h[perm[i] * stride] = perm[(i + 1) % nodes] * (unsigned)stride;
        unsigned* d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        int hops = 200000;
        float ms = time_ms([&]{ k_chase<<<1, 1>>>(d, hops, perm[0] * (unsigned)stride, sink); }, 3);
        printf("  %4zu MB working set: %6.0f ns/load\n", mb, ms * 1e6 / hops);
        CK(cudaFree(d));
    }

    printf("\n[6] sync costs\n");
    if (coop) {
        for (int nb : {sms, 2 * sms}) {
            int iters = 20000; void* args[] = {&iters, &sink};
            float ms = time_ms([&]{ CK(cudaLaunchCooperativeKernel((void*)k_barrier, nb, 256, args)); }, 3);
            printf("  grid barrier, %2d CTAs (cooperative): %6.2f us each\n", nb, ms * 1e3 / iters);
        }
    }
    for (int tiny = 0; tiny < 2; tiny++) {
        const int N = 500; float* x; CK(cudaMalloc(&x, sms * 256 * 4));
        cudaStream_t s; CK(cudaStreamCreate(&s)); cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < N; i++) { if (tiny) k_tiny<<<sms, 256, 0, s>>>(x); else k_empty<<<1, 32, 0, s>>>(); }
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        float ms = time_ms([&]{ CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s)); }, 5);
        printf("  CUDA graph, %d dependent %s nodes: %5.2f us per node\n", N,
               tiny ? "tiny (8x256)" : "empty (1x32) ", ms * 1e3 / N);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s)); CK(cudaFree(x));
    }
    return 0;
}
