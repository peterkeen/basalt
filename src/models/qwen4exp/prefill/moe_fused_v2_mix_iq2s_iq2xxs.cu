// src/models/qwen4exp/prefill/moe_fused_v2_mix_iq2s_iq2xxs.cu - v2 mixed gate/up kernel (IQ2_S gate, IQ2_XXS up).
#include "moe_fused_v2.cuh"

namespace basalt::prefill::fused {
namespace {
using C = v2_detail::V2<v2_detail::T_IQ2_S, true, v2_detail::T_IQ2_XXS>;
__global__ void __launch_bounds__(kV2Threads, 1)
v2_kernel_mix_iq2s_iq2xxs(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                          const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm,
                          int* __restrict__ ctr) {
    v2_detail::v2_stage<v2_detail::T_IQ2_S, true, v2_detail::T_IQ2_XXS>(
        (int) threadIdx.x, b, geo, tb, act, src, out, dm, ctr);
}
}  // namespace
V2Spec v2_spec_mix_iq2s_iq2xxs() { return {(const void*) v2_kernel_mix_iq2s_iq2xxs, C::SMEM, C::NFB}; }
}  // namespace basalt::prefill::fused
