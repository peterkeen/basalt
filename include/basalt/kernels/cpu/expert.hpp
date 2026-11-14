// include/basalt/kernels/cpu/expert.hpp - the CPU expert path (Q2_0 experts).
//
// WHAT MAKES IT FAST:
//
//   * `vpdpbusd` (AVX512-VNNI) does four int8 MACs per instruction, which keeps the kernel memory-bound
//     instead of compute-bound.
//   * `vpmultishiftqb` (AVX512-VBMI) unpacks the 2-bit codes eight at a time.  Its ARGUMENT ORDER is
//     (control, data) and was determined empirically - the Intel guide states the opposite, and following the
//     guide produced 43 of 64 wrong codes.
//   * The row accumulator stays a FLOAT VECTOR and is reduced ONCE at the end of the row.  Reducing per
//     64-weight block costs more than the dot products themselves and makes the kernel compute-bound.
//
// THE CONTRACT THAT MATTERS, because collapsing it is invisible and cannot match ggml's logits: one Q2_0
// WEIGHT block is QK=64 weights with one fp16 scale, while one ACTIVATION chunk is QKA=32 elements with
// its OWN scale.  So a weight block spans TWO activation chunks with two different scales.  Using one scale
// per weight block is the natural-looking mistake.
#pragma once

#include <cstdint>
#include <cstddef>

namespace basalt::kernels::cpu {

// ---- geometry, all of it fixed by the artifact (every Q2_0 tensor has QK=64) ----
inline constexpr int QK = 64;       // QK2_0: weights per fp16 scale
inline constexpr int QKA = 32;      // QK8_1/QK8_0: ACTIVATION elements per scale
inline constexpr int BB = 18;       // bytes per 64-weight Q2_0 block: 2 fp16-scale + 16 codes
/// The widest activation one quantized block holds.
inline constexpr int kActMaxWidth = 2560;
inline constexpr int MAXC = kActMaxWidth / QKA;   // activation chunks across the widest reduction

// ---- the blob's internal layout (one expert) ----

/// One activation in the engine's planar storage: `QKA` elements per chunk.
/// The quantizer uses FP32 scales and half-away rounding; it is not the ggml CPU Q8_0 contract.
///
/// The historical function name `act_quant_q8_1` does not describe a native
/// ggml block layout. One 64-weight block always spans two 32-element chunks.
///
/// `hx[k] = scale[k] * sum[k]` is the weight-independent correction the Q2_0 identity needs
/// (`sum (c-1) d_w xhat d_x = d_w (d_x*sum(c*xhat) - d_x*sum(xhat))`), precomputed once per layer rather than
/// once per row: 640 rows would otherwise recompute it 640 times.
struct ActQ {
    alignas(64) int8_t q[kActMaxWidth];
    float scale[MAXC];
    int32_t sum[MAXC];
    float hx[MAXC];
    int nchunks;
};

/// `x` -> `a` in the configured contract. The function name is historical;
/// `n` must be a multiple of QKA and at most kActMaxWidth. No allocation occurs.
void act_quant_q8_1(const float* x, int n, ActQ& a);

inline constexpr int MAXT = 8;

/// Rows [r0, r1) of a Q2_0 matrix in the GGUF block layout (18 bytes per 64 weights, `row_bytes`
/// per row, `nblocks` blocks each) against `nt` activations: out[t][r].
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1);
/// The same two for CPUs without AVX-512 (src/kernels/cpu/q2_avx2.cpp, compiled for AVX2 only).  The rows take the
/// AVX-VNNI build where the CPU has it; `_base` and `_vnni` name the two builds for the tests (every output bit is the same).
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1);
void q2_0_gguf_rows_multi_avx2_base(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                    float* const* out, int r0, int r1);
void q2_0_gguf_rows_multi_avx2_vnni(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                    float* const* out, int r0, int r1);
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a);

}  // namespace basalt::kernels::cpu
