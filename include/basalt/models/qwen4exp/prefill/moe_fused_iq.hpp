// include/basalt/models/qwen4exp/prefill/moe_fused_iq.hpp - the fused int8 prompt experts (moe_fused.hpp) for the native GGUF
// experts (an expert is its raw GGUF slices [gate rows | up rows | down rows], formats per layer in
// expert_layout().fmt), used wherever available (devices with int8 MMA; others use MMQ).
//
// The grouping on the GPU (fused::group), the per-expert pointers (fused::Batch: a cache slot or a ring slot, no
// gather) and the launch shape are the Q2_0 path's.  Raw GGUF weight bytes arrive in shared memory via 16-byte
// cp.async and are decoded straight into the int8 mma A fragments, over up to 128 routed rows per work item.  The
// activations and H are int8 per 32 values in natural order (quantize_act_native).
//
// Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (and mixed IQ2_S gate + IQ2_XXS up); down Q2_0,
// IQ4_NL - every layer of the IQ2_XS, IQ3_XXS and IQ3_S packs except IQ1_M (which MMQ does not cover either).  Other
// layers keep MMQ.
#pragma once

#include "basalt/models/qwen4exp/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::fused {

/// One layer's native expert geometry (cpu::NativeFmt's fields the kernels read; n_embd 2560, n_ff 640).
struct NativeGeom {
    int gu_type = -1, d_type = -1;    ///< ggml types
    size_t gu_row = 0, d_row = 0;     ///< bytes per gate row and per down row
    size_t up_off = 0, down_off = 0;  ///< inside the blob
    int up_type = -1;                 ///< the up rows' type and row bytes (-1: the gate's)
    size_t up_row = 0;
};

/// enabled() (int8 MMA), and the kernels cover this gate/up and down pair on this device (`up_type`: the up rows'
/// type, -1 for gu_type's; a mix is covered for the pairs the kernels are built for).
bool native_supported(int gu_type, int d_type, int up_type = -1);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)): int8 per 32 values in natural order, the native kernels' form.
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// experts() for native blobs: b.blob[e - e0] is the native blob of expert e (gate at 0, up at g.up_off, down at
/// g.down_off; 2-byte aligned).  `xa` from quantize_act_native; H in `ha` (act_bytes(n, 640)); down into `dm` at the
/// rows of `group`.
void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream);

}  // namespace basalt::prefill::fused
