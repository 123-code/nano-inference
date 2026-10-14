# minfer

**A from-scratch CUDA inference engine for Qwen2.5-1.5B that beats llama.cpp on a Jetson Orin Nano Super — ~1.5× out-of-the-box llama.cpp, 1.16× a fully tuned build — and the rigorous, measured optimization journey (dead ends included) that got it there.**

No deep-learning framework, no inference library. Just a GGUF loader, hand-written CUDA kernels, and a CPU reference oracle that every GPU change is validated byte-for-byte against. The point isn't to replace llama.cpp or Ollama — it's to find out how close to the silicon's *physical limit* you can push single-stream decode on an 8 GB embedded GPU, and to be honest about what worked and what didn't.

## Results

Qwen2.5-1.5B-Instruct (q4_0), greedy decode tg128, Jetson Orin Nano Super (sm_87, 8 GB), `jetson_clocks` pinned, all runs back-to-back (2026-07-07):

| Config | tok/s |
|---|--:|
| llama.cpp, stock q4_0 GGUF, `-fa 0` (out-of-the-box) | 45.5 |
| llama.cpp master b9899, `-fa 1` + q4_0 output tensor (best tuned) | 60.1 |
| **minfer** (same all-q4_0 model file) | **69.6** |

**A note on honesty:** an earlier draft claimed 1.8× over llama.cpp (baseline "37.7"). That baseline turned out to be measured at unpinned memory clocks — the same fairness checks that killed the headline also found two free llama.cpp speedups on this box (flash attention now works on sm_87: +10%; the stock Q6_K output tensor is slow here, requantizing it to q4_0: +17%). The corrected claim: **~1.5× over llama.cpp as shipped, 1.16× over llama.cpp with every tuning lever pulled** — same model file, same clocks, back-to-back.

The optimization journey (each step A/B-tested against its own same-session baseline):

| Milestone | tok/s | note |
|---|--:|---|
| M2 — naive CUDA port | 3.0 | 1 block/row scalar GEMV |
| M3 — warp-per-row GEMV | 9.1 | fixed uncoalesced activation reads |
| M3.5 — `__dp4a` INT8 GEMV | 10.2 | int8 activations + INT8 MACs |
| M5 — fused attention | 25.5 | one block/head, block softmax |
| M6 — int8 lm_head | 45.1 | past out-of-the-box llama.cpp |
| M9 — SIMD nibble unpack | 50.0 | q4 GEMV was unpack-ALU-bound |
| M10 — wide (uint4) loads | 59.0 | matmuls were L1-transaction-bound |
| M12 — split-KV attention | +6–9% at depth | flash-style, fixes attention occupancy |
| M14 — q4 lm_head | 69.6 | ~60% of the real memory roofline (see M15) |
| M15 — one persistent kernel | ~95 | whole token loop in one cooperative kernel; GEMVs at ~100 GB/s |
| M16 — tensor-core multi-token steps | — | an 8-token step costs 1.52× a 1-token step; logits bit-identical for any step size |
| **M17 — lossless n-gram speculation** | **95–179** | **output identical to plain decode; +24% on code/lists, +89% on structured rewrites** |

The headline number isn't the interesting part — the measurement discipline is. Decode is memory-bound: every token streams the weights once, and `tok/s = bandwidth ÷ bytes-per-token`. **M14 concluded the engine was at ~90% of the roofline and decode was "essentially solved" — that was wrong twice over** (M15): the 75.2 GB/s "roofline" came from a benchmark dominated by launch overhead (6.88 MB per launch), the real streaming limit of this box is **101 GB/s** (99% of the 102.4 GB/s LPDDR5 peak, `bench/roofline.cu`), and since the q4 lm_head the engine reads **0.868 GB/token**, not ~0.99. So M14 was at ~60% of the roofline. M15 replaces the 567-kernel CUDA graph with one persistent kernel (~95 tok/s plain decode; its GEMVs stream at ~100 GB/s, the rest of the step is attention, norms and barriers); the ceiling for this model file is ~116 tok/s. M16–M17 then go *past* the bandwidth wall the only way single-stream decode can: verify several drafted tokens per weight pass.

## What this is / isn't

**Is:** a single-file CUDA engine (one model, one GPU, batch-1) built to study inference optimization on a memory-bound embedded GPU, with a strict correctness gate and profile-driven decisions at every step.

**Isn't:** a general runtime. No batching/serving, no multi-GPU, no broad model/quant support, no Python API. For production use on a Jetson, use llama.cpp or Ollama. ([How it compares to TensorRT-LLM](#how-it-compares-to-tensorrt-llm) is covered below — short version: their edge is *throughput via batching*, not batch-1 latency, where minfer is already near the same wall.)

## Quickstart

Requires CUDA + an `sm_87` device (Jetson Orin). You provide the model — download `qwen2.5-1.5b-instruct-q4_0.gguf` (e.g. from the Qwen GGUF repos) and point `MODEL` at it.

```bash
cmake -S . -B build && cmake --build build -j6
./tools/compare.sh                 # correctness gate: greedy tokens match llama.cpp byte-for-byte

# watch text stream live:
MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf ./tools/run.sh "Write a haiku about the sea."

# faster on grounded/repetitive output (lossless n-gram speculative decode):
MODEL=... SPEC=1 ./tools/run.sh "List the first 20 prime numbers."

# M15–M17 persistent-kernel engine (same CLI as cuda_infer; n-gram speculation on by default, SPEC=0 to disable):
RAWIDS=1 ./build/mega $MODEL 256 <ids...>
tools/specbench.sh      # cuda_infer vs mega, plain and speculative, + checks spec output == plain output
tools/ab.sh             # A/B speed + teacher-forced agreement vs cuda_infer

# steady-state benchmark (token-ID output, no early stop):
MODEL=... ./tools/speed.sh        # -> minfer-cuda decode: ... tok/s
```

`cuda_infer` detokenizes (GPT-2 byte-level BPE) and streams text per token, stopping at `<|im_end|>`.

**Env knobs:** `SPEC=1` n-gram speculative decode (lossless) · `LMHEAD=0` byte-exact fp32 path (slower, for the gate) · `LMHEAD=4` q4 lm_head (default) · `KVQ=1` int8 KV cache · `ATTN=0` monolithic attention · `RAWIDS=1` print token IDs.

## How it works

- **`src/gguf.h`** — header-only GGUF v3 mmap reader + dequant (Q4_0/Q4_1/Q8_0/Q6_K/F16/F32). Per-tensor `madvise(MADV_DONTNEED)` after upload so the mmap + device copy don't double-count memory.
- **`src/cpu_infer.cpp`** — CPU fp32 reference forward. The oracle: every GPU kernel is validated to match it (and llama.cpp) in token-ID space.
- **`src/cuda_infer.cu`** — the GPU engine. Weights stay quantized-resident (~1 GB). The decode step is captured as a **self-advancing CUDA graph** (token/pos in device memory, device-side argmax + `pos++`). Hot kernels:
  - **`gemv_dp4a`** — warp-per-row q4_0 GEMV with int8 activations, `__dp4a` MACs, SIMD nibble unpack, and `uint4` wide loads.
  - **`attn_partial` / `attn_combine`** — flash-style split-KV attention (online softmax) for high occupancy at depth.
  - **`gemv_q8_dp4a`** — coalesced int8 lm_head.
- **`src/mega.cu`** — M15–M17: the same model as one persistent cooperative kernel (8 CTAs × 512 threads): packed 32-row-group q4 weights in one arena, int8 activations in shared memory, grid barriers between phases, on-device argmax, token loop and n-gram drafting; the host only drains a mapped ring buffer. A step carries up to 8 tokens (prompt chunks, speculative verify): 1 token runs `__dp4a`, more run `mma.m16n8k32` tensor cores over the same weights, with every float operation explicitly rounded so a token's logits do not depend on the step size.
- **Speculative decode** — n-gram / prompt-lookup drafting, verified by a batched multi-token forward; lossless (commit a draft token only if it equals the true greedy argmax), with an adaptive gate so it never regresses.

## The optimization journey

The interesting part. Each step is profile-driven and A/B-tested against the same thermal/clock state. **The dead ends are included on purpose** — knowing *why* an idea doesn't pay off on this box is the actual result.

<details>
<summary><b>Full milestone log (M0 → M14)</b></summary>

- **M0** — llama.cpp decode-vs-depth baseline + roofline. Decode ~flat vs context at 1.5B (`-fa 0` barely costs here → fused-attn is a 7B story).
- **M1** — GGUF loader + CPU fp32 reference. Greedy matches llama.cpp byte-for-byte (`tools/compare.sh`): validates GGUF parse, Q4_0/Q6_K dequant, RMSNorm, QKV+bias, NEOX RoPE, GQA causal attention, SwiGLU, untied lm_head.
- **M2** — naive CUDA port. **3.0 tok/s.** Weights kept quantized-resident (fp16 expansion won't fit: 3.6 GB cudaMalloc → NvMap ENOMEM).
- **M3** — warp-per-row GEMV, lane = element-within-block. **9.1 tok/s.** Key finding via nsys: the bottleneck was **uncoalesced activation reads, not weights**.
- **M4** — decode captured as a self-advancing **CUDA graph**. Throughput a wash (9.05→9.12) — we're kernel-bound, not launch-bound. Kept as infrastructure (pays off once kernels shrink).
- **M3.5** — `__dp4a` INT8 Q4_0 GEMV (int8 activations). **10.2 tok/s, exact match.** Layer matmuls near 58 GB/s.
- **M5** — `attn_head`: one block/head, shared-q + block softmax. Kernel 60× faster (attn 55%→2%). **25.5 tok/s.** (Use `expf` not `__expf` — fast-math flips a borderline token.)
- **M6 — int8 lm_head (the 51% bottleneck).** Scalar Q6_K dequant was compute-bound at ~9 GB/s. Two findings: a faithful Q6_K-dp4a kernel only hit 31.9 (scattered ql/qh can't coalesce); **requantizing the lm_head Q6_K→Q8_0 at load** gives a linear layout → coalesced `gemv_q8_dp4a` → **45.1 tok/s, past out-of-the-box llama.cpp.** Accuracy: int8 lm_head matches the fp32 oracle ~6 tokens (the divergence is the *activation* int8 quant, not the weights — same strategy llama.cpp uses). `LMHEAD=0` keeps the byte-exact path for the gate.
- **M7 — speculative decoding (n-gram), lossless.** Built the verify primitive (batched multi-token forward). **Key finding (`BENCHT=1`):** batched cost ≈ 19 + ~13·T ms — each extra verify token ≈ 0.6× a decode step, NOT free (int8 weights make decode compute-*balanced*). So a **draft model (0.5B) loses here** → n-gram (free draft) is the viable path. Lossless via greedy-match commit + an adaptive gate. Repetitive output **+37–45%**, prose +6–13%, code ~break-even.
- **M8 — templated batched kernels.** `template<int T>` accumulators cut the fixed per-pass cost (`19.3+12.9·T` → `10.7+13.5·T` ms). Code-gen SPEC now firmly break-even.
- **M9 — profile-guided GEMV tuning.** nsys: **89% of decode is the two matmuls.** The q4 GEMV ran at ~43 GB/s while the unpack-free lm_head hit ~59 → q4 was **unpack-ALU-bound, not memory-bound.** Fix: SIMD nibble masking (`word & 0x0F0F0F0F`) + folded −8 offset (`Σ(n−8)x = Σn·x − 8·Σx`). Bit-exact. **44 → 50 tok/s.**
- **M10 — wide loads (ncu-guided).** ncu: the big FFN matmuls are **L1/TEX-bound at ~95%**, L2 only ~56%, compute 20–48% — the load-unit/L1 transaction rate is the limit, *not* DRAM. Fix: load activations (and Q8 lm_head) as `uint4` (2 wide loads instead of 8). **50 → 59 tok/s.**
- **M11 — int8 KV cache: WASH (negative result).** K/V int8 + fp16 scale per (token,kv_head), 4× fewer KV bytes, near-lossless. *Hypothesis: flatten the long-context slowdown.* It didn't. Profiled why: at pos=861 the KV cache is only **~9% of per-step bytes**, and attention is **~2–3% of decode time** — so cutting KV bytes can't move the needle, and the dequant overhead makes it ~1–2 tok/s *slower*. Kept opt-in (`KVQ=1`) for cache *capacity*. **Lesson: profile before optimizing.**
- **M11.5 — root-caused the long-context slide: attention occupancy, not the matmuls.** Controlled nsys at fixed pos: per-call **median `gemv_dp4a` is flat** (pos-independent) while **`attn_head` grows 5.5× (23→129 µs)**. Cause: `attn_head` launches only **12 blocks = 48 warps** vs the Orin's **8 SMs × 48 = 384-warp capacity → 12.5% occupancy.** At high pos the O(pos) serial loops dominate with 88% of the GPU idle.
- **M12 — split-KV (flash-style) attention.** `attn_partial` (one block per head×chunk, local softmax) + `attn_combine` (online-softmax merge) → ~5× more blocks fills the SMs. **Bit-exact 20/20.** Per-call attention −3.7× in *cycles*; end-to-end **+6–9% at depth** (the matmuls still cap it — attention is only ~17% of the step even at pos~805). A **two-graph pos-gated dispatch** (monolithic below pos 256, split beyond) avoids the −2% short-context cost.
- **M13 — multi-step decode graph: WASH (and it corrected the record).** Batched K self-advancing launches with no per-token host sync. A/B (`MSTEP=1` vs `32`, short *and* sustained) is **dead flat** → the per-token sync is ~0.3% of a memory-bound step. **So decode is memory-bandwidth (EMC-clock) bound, NOT host-issue-bound** — the earlier "clock pinning" win was the memory clock, not host latency. Default `MSTEP=1` (smooth streaming).
- **M14 — measured the roofline + q4 lm_head (+4.5%, free).** A streaming-read microbench measures **75.2 GB/s achievable** (the `cudaDeviceProp` "peak" of 32.6 GB/s is bogus on Tegra unified memory). minfer reads ~0.99 GB/token → **~90% of the roofline, already near the limit.** Only two levers remain: raise achieved BW (≤~18% left, hard) or cut bytes/token. First cut: the **lm_head was ~25% of the bytes** and had been bloated to Q8_0 (248 MB) for coalescing — requantize it to **q4_0 (131 MB, now default)**. Same accuracy → **+4.5% (65→68.5 tok/s).** *Correction (M15): both the roofline and the bytes/token here were wrong — see M15.*
- **M15 — the whole decode loop in one persistent kernel (`src/mega.cu`).** First, re-measured the roofline properly (`bench/roofline.cu`: one launch over 768 MB, several 16 B loads in flight per thread): **101 GB/s** for any sane streaming pattern — even 2 SMs alone pull 99 GB/s. Then broke the M14 step down by kernel (`bench/breakdown.cu`, real kernels, synthetic weights): GEMVs **11.5 ms at 75 GB/s** (warp-per-row wastes lanes on 1536-wide rows: gate/up 75 GB/s, k/v 39 GB/s; only `ffn_down` with its 8960-wide rows hit 90) + **368 tiny kernels = 1.8 ms** with DRAM idle. Fix: one cooperative kernel (1 CTA/SM, grid barrier 1.2 µs instead of ~1–2 µs per graph node) runs every layer of every token; weights repacked at load into 32-row groups (a warp reads 512 contiguous bytes, each lane owns a row — no shuffle reductions); the activation is quantized once into shared memory; GQA-grouped split-KV attention (one K/V read serves all 6 q heads); argmax, embedding and the token loop stay on the GPU. **~95 tok/s vs 69–70 back-to-back (+38%), 88 vs 50 tok/s averaged over 2048 tokens.** Same numerics as cuda_infer (identical per-block integer sums) up to float reassociation: teacher-forced top-1 agreement 99.4% on a 5-prompt suite, every miss a near-tie (≤0.17 logit) — cuda_infer agrees with *itself* 99.8% when only its attention order changes (`tools/ab.sh`). Dead end on the way: prefetching the next matrix into L2 during attention cost more than it saved.
- **M16 — multi-token steps on tensor cores, bit-identical to single-token steps.** Decode is memory-bound, so a step that streams the weights once for *several* tokens is almost free — if the math keeps up. `__dp4a` doesn't (cuda_infer's batched verify cost `10.7 + 13.5·T` ms), so for T > 1 the same packed q4 weights feed `mma.m16n8k32.s8` (nibbles unpacked with one AND/shift, tokens as the N dimension). Both paths form the same per-block integer sums and accumulate them through the same explicitly rounded expression, and every other phase (norms, attention split and merge order, residuals) processes each token exactly as it would alone — so **a token's logits are bit-identical whether it runs alone or in a step of 8** (verified by hashing every logit margin under teacher forcing, `TEACHER=k TCHUNK=1/2/4/8`). Step cost: T = 1/2/4/8 → 10.5/11.5/13.0/15.9 ms (**8 tokens for 1.52× the price of 1**). Getting there was mostly a fight with the compiler, not the hardware: shuffles inside warp-divergent code compile to call+WARPSYNC helpers that serialize everything (fixed with block-uniform trip counts and selects), runtime loop bounds read from constant memory blocked all ILP across attention heads, shared-memory pointers passed into out-of-line functions became generic loads, and conditionally-filled register arrays stayed live across the whole layer loop (1.5 KB of spills).
- **M17 — lossless n-gram speculative decoding, on the GPU.** Each CTA keeps the token history in shared memory and runs the same prompt-lookup search (longest n-gram ≤ 3, latest match), so drafts need no extra barrier; a verify step runs `[current, draft₁..draft_K]` and commits up to and including the first prediction that disagrees with its draft. K is chosen per step to maximize expected tokens per unit of step cost from a running acceptance estimate and the measured cost curve, so speculation never loses. Because of M16, the output is **token-for-token identical to plain greedy decode** (checked on every prompt by `tools/specbench.sh`). 256 tokens/prompt: prose 95→97 tok/s (1.04 tokens/step), Python+tests 95→118, a list 95→117, list→JSON rewrite 95→**179** (2.3 tokens/step) — vs cuda_infer's 69–70 tok/s, whose own n-gram SPEC gains only 1–2% here because its verify pass is too expensive.

</details>

### What didn't work (and why that's useful)

- **int8 KV cache** — a wash. KV is ~9% of per-step bytes and attention is ~2–3% of decode time; you can't speed up the whole by shrinking a tiny slice.
- **Multi-step decode graph** — a wash. Removing per-token host sync does nothing because decode is *memory-bandwidth-bound*, not host-bound (this also corrected an earlier wrong diagnosis).
- **Draft-model speculative decoding** — loses on this box. INT8 weights make batched verify compute-balanced, so verify-of-K isn't free; only *free-draft* n-gram speculation pays off.
- **Lower-precision "fast-math" / fp16 scale tricks** — flipped borderline greedy tokens or broke coalescing; abandoned.

The recurring lesson: on a memory-bandwidth-bound workload, only two things matter — bytes-per-token and achieved bandwidth. Every win came from one of those; every wash ignored them.

## How it compares to TensorRT-LLM

TensorRT-LLM's throughput comes from **continuous batching + paged KV cache** — amortizing the weight read across many concurrent requests (~N× aggregate). That's a *throughput* story, not a *batch-1 latency* story. For one user on one GPU, minfer hits the same memory wall TensorRT-LLM would. Their other advantages (fp8/fp4, tensor-core GEMMs, Eagle/Medusa spec decode, multi-GPU) are mostly moot on `sm_87` or orthogonal to single-stream latency. Building the batching path would be the way to chase aggregate throughput — but on 8 GB you'd only fit a handful of sequences, so it's not pursued here.

## Reproducibility notes

- **Hardware:** Jetson Orin Nano Super, `sm_87`, 8 GB unified LPDDR5. Achievable read bandwidth ~75 GB/s (measured). Numbers won't transfer to other GPUs.
- **Clocks:** run `sudo jetson_clocks` first. The memory (EMC) clock dominates decode tok/s and unpins when the board sleeps — absolute numbers swing ~40–68 tok/s with it, so all optimizations are validated by **back-to-back A/B**, never raw absolutes.
- **Correctness gate:** `tools/compare.sh` (cpu_infer vs llama.cpp oracle) and `LMHEAD=0` (cuda vs cpu_infer, bit-exact 20/20). `tools/ref_oracle.cpp` links prebuilt `libllama` for reference tokens.
- **Model details:** Qwen2.5-1.5B — 28 layers, d=1536, ff=8960, 12 Q / 2 KV heads, hd=128, RoPE base 1e6, rms_eps 1e-6, vocab 151936, untied Q6_K lm_head, QKV biases.

## License

MIT — see [LICENSE](LICENSE). Qwen2.5 model weights are licensed separately by their authors; download them yourself.
