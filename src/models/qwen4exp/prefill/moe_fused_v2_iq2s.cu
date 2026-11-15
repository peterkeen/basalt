// src/models/qwen4exp/prefill/moe_fused_v2_iq2s.cu - v2 gate/up kernel for IQ2_S.
#include "moe_fused_v2.cuh"

namespace basalt::prefill::fused {
namespace {
using C = v2_detail::V2<v2_detail::T_IQ2_S, true>;
__global__ void __launch_bounds__(kV2Threads, 1)
v2_kernel_iq2s(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
               const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm,
               int* __restrict__ ctr) {
    v2_detail::v2_stage<v2_detail::T_IQ2_S, true>((int) threadIdx.x, b, geo, tb, act, src, out, dm, ctr);
}
}  // namespace
V2Spec v2_spec_iq2s() { return {(const void*) v2_kernel_iq2s, C::SMEM, C::NFB}; }
}  // namespace basalt::prefill::fused
