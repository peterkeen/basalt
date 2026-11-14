// include/basalt/kernels/cpu/kq_avx512.hpp - AVX-512 VNNI multi-token dot products for K-quant and
// legacy expert formats (Q4_K, Q5_K, Q6_K, Q5_1, Q8_0), bit for bit equal to ggml's single-token dots.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

bool kq512_supported(int ggml_type) noexcept;
bool kq512_fits(int ggml_type, int n) noexcept;

/// ff[t][r] = silu(g) * u for g, u the gate and up dots, rows [r0, r1); `scaled` multiplies g, u by sg, su first;
/// limit > 0 caps g above and clamps u to +-limit (the same expressions as the ggml path).
void kq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                   bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1);

/// out[t][r] = w_r . a[t] (multiplied by `sd` when `scaled`), rows [r0, r1).
void kq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, float sd, bool scaled,
                const void* const* act, int nt, float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
