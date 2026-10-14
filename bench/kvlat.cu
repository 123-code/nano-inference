// bench/kvlat.cu — what does one batch of attention K/V loads cost on this Orin?
// Mirrors mega's attention: 8 CTAs (1 per SM), WARPS warps each issue batches of NL independent
// loads (one 128 B line per warp-instruction), consume them, then issue the next batch.
// Variants: load flavour (.cg strong vs weak L1::no_allocate) x line stride (consecutive lines,
// 1 KB stride like K stored dim-major, 4 KB stride).
//   nvcc -O3 -arch=sm_87 -o build/kvlat bench/kvlat.cu && ./build/kvlat
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} } while(0)

template<int MODE> __device__ __forceinline__ float ld(const float* p) {
    float v;
    if (MODE == 0) asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    else if (MODE == 1) asm volatile("ld.global.L1::no_allocate.f32 %0, [%1];" : "=f"(v) : "l"(p));
    else asm volatile("ld.global.relaxed.gpu.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
template<int MODE, int NL>
__global__ void k(const float* buf, size_t stride_f, int nbatch, unsigned long long* out, float* sink) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // each warp walks its own region so nothing is shared through L2 across warps/SMs
    const float* base = buf + ((size_t)(blockIdx.x * blockDim.x / 32 + warp) * nbatch * NL) * stride_f + lane;
    float acc = 0.f;
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int b = 0; b < nbatch; b++) {
        float v[NL];
        #pragma unroll
        for (int i = 0; i < NL; i++) v[i] = ld<MODE>(base + (size_t)(b * NL + i) * stride_f);
        #pragma unroll
        for (int i = 0; i < NL; i++) acc = fmaf(acc, 1.0001f, v[i]);
    }
    unsigned long long t1 = clock64();
    if (lane == 0) out[blockIdx.x * (blockDim.x / 32) + warp] = t1 - t0;
    if (acc == 12345.f) sink[threadIdx.x] = acc;
}

int main() {
    const size_t bytes = 512ull << 20;
    float* buf; CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 0, bytes));
    unsigned long long* out; CK(cudaMalloc(&out, 64 * 8));
    float* sink; CK(cudaMalloc(&sink, 4096));
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("per-batch latency (clock64 cycles at %d MHz attr; ns assumes that clock)\n", clk / 1000);
    printf("%-28s %6s %6s %10s %8s\n", "flavour", "warps", "stride", "cyc/batch", "ns");
    for (int mode = 0; mode < 3; mode++)
        for (int warps : {1, 2, 16})
            for (int stride : {32, 256, 1024}) {      // floats: 128 B (consecutive lines), 1 KB, 4 KB
                const int nb = 64;
                auto run = [&]() {
                    if (mode == 0) k<0, 32><<<8, 32 * warps>>>(buf, stride, nb, out, sink);
                    else if (mode == 1) k<1, 32><<<8, 32 * warps>>>(buf, stride, nb, out, sink);
                    else k<2, 32><<<8, 32 * warps>>>(buf, stride, nb, out, sink);
                };
                if ((size_t)8 * warps * nb * 32 * stride * 4 > bytes) continue;
                run(); CK(cudaDeviceSynchronize()); run(); CK(cudaDeviceSynchronize());
                unsigned long long h[64]; CK(cudaMemcpy(h, out, 8 * warps * 8, cudaMemcpyDeviceToHost));
                double avg = 0; for (int i = 0; i < 8 * warps; i++) avg += h[i]; avg /= 8 * warps * nb;
                const char* nm[3] = {"ld.global.cg (strong)", "ld.global.L1::no_allocate", "ld.global.relaxed.gpu"};
                printf("%-28s %6d %5dB %10.0f %8.0f\n", nm[mode], warps, stride * 4, avg, avg * 1e6 / clk);
            }
    return 0;
}
