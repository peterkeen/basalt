// src/models/qwen4exp/prefill/gemm_wht128.hpp - WHT-128 int8 tensor-core path for Gemm::native.
#pragma once

#include <cstdint>

namespace basalt::prefill {

void gemm_wht128_set_override(int mode);

bool gemm_wht128_native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                        int64_t K, int64_t ldy, float beta, uint16_t* scratch, int64_t scratch_elems, void* stream);

}  // namespace basalt::prefill
