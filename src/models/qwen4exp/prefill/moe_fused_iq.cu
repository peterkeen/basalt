// src/models/qwen4exp/prefill/moe_fused_iq.cu - see include/basalt/models/qwen4exp/prefill/moe_fused_iq.hpp.
//
// Activation quantization and host dispatch for the native fused prompt expert kernels (each format variant's kernel
// lives in its own moe_fused_v2_*.cu unit, sharing the device stage in moe_fused_v2.cuh).
#include "basalt/models/qwen4exp/prefill/moe_fused_iq.hpp"
#include "moe_fused_v2.hpp"

#include <cuda_runtime.h>

#include "basalt/ggml/ggml.h"
#define GGML_COMMON_DECL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include "basalt/log.hpp"

namespace basalt::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill fused experts (native): %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;   // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused

constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18, "the block layouts the native kernels decode");

size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

__device__ __forceinline__ void quant_act_nat_stage(int64_t w, int lane, const float* __restrict__ x, int64_t nblk,
                                                    uint8_t* __restrict__ xa) {
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    out[lane] = (uint8_t) (int8_t) (a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0);
    out[32 + lane] = (uint8_t) (int8_t) (a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0);
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, 0.0f, 0.0f);
}

__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    quant_act_nat_stage(w, threadIdx.x & 31, x, nblk, xa);
}

unsigned blocks(int64_t work, int per_block) { return (unsigned) std::max<int64_t>(1, (work + per_block - 1) / per_block); }

struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ = 1;
};
std::mutex g_mu;
DevInfo g_dev[32];

bool setup_v2_spec(const V2Spec& sp, int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, sp.fn) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(sp.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sp.smem) != cudaSuccess) return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, sp.fn, kV2Threads, sp.smem) != cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (major < 8) return d;
    int occ = 1 << 20;
    if (setup_v2_spec(v2_spec_iq2xxs(), occ) && setup_v2_spec(v2_spec_iq2xs(), occ) &&
        setup_v2_spec(v2_spec_iq2s(), occ) && setup_v2_spec(v2_spec_iq3xxs(), occ) &&
        setup_v2_spec(v2_spec_iq3s(), occ) && setup_v2_spec(v2_spec_iq4xs(), occ) &&
        setup_v2_spec(v2_spec_mix_iq2s_iq2xxs(), occ) &&
        setup_v2_spec(v2_spec_q2_0(), occ) && setup_v2_spec(v2_spec_iq4nl(), occ)) {
        d.ok = true;
        d.occ = occ;
    }
    cudaGetLastError();
    return d;
}

int* v2_counters(const void* scratch, int64_t n, int n_expert) {
    const size_t tiles = (size_t) (n + kTileRows - 1) / kTileRows + (size_t) n_expert;
    const size_t at = (tiles_at(n_expert) + tiles * sizeof(int2) + 15) / 16 * 16;
    return (int*) ((uint8_t*) const_cast<void*>(scratch) + at);
}

bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS;
}
bool d_covered(int t) { return t == T_Q2_0 || t == T_IQ4_NL; }
bool mix_covered(int tg, int tu) { return tg == T_IQ2_S && tu == T_IQ2_XXS; }

V2Spec pick_v2_gu(int tg, int tu) {
    if (tu != tg) return v2_spec_mix_iq2s_iq2xxs();
    switch (tg) {
        case T_IQ2_XXS: return v2_spec_iq2xxs();
        case T_IQ2_XS:  return v2_spec_iq2xs();
        case T_IQ2_S:   return v2_spec_iq2s();
        case T_IQ3_XXS: return v2_spec_iq3xxs();
        case T_IQ3_S:   return v2_spec_iq3s();
        default:        return v2_spec_iq4xs();
    }
}

V2Spec pick_v2_d(int td) { return td == T_Q2_0 ? v2_spec_q2_0() : v2_spec_iq4nl(); }

void launch_v2(const V2Spec& sp, unsigned grid, Batch b, NativeGeom g, Tables tb, const void* act,
               const int32_t* src, void* out, float* dm, int* ctr, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    void* args[] = {&b, &g, &tb, &a, (void*) &src, &o, &dm, &ctr};
    ck(cudaLaunchKernel(sp.fn, dim3(grid), dim3(kV2Threads), args, (size_t) sp.smem, s), "launch_v2");
}

}  // namespace

bool native_supported(int gu_type, int d_type, int up_type) {
    if (up_type >= 0 && up_type != gu_type)
        return requested() && mix_covered(gu_type, up_type) && d_covered(d_type) && dev_info().ok;
    return requested() && gu_covered(gu_type) && d_covered(d_type) && dev_info().ok;
}

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act_native");
}

void experts_native(const Batch& b, const NativeGeom& g_in, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    NativeGeom g = g_in;
    if (g.up_type < 0) {
        g.up_type = g.gu_type;
        g.up_row = g.gu_row;
    }
    const bool mix = g.up_type != g.gu_type;
    if (!d.ok || !(mix ? mix_covered(g.gu_type, g.up_type) : gu_covered(g.gu_type)) || !d_covered(g.d_type)) {
        basalt::log::error("basalt", "prefill fused experts (native): types %d / %d / %d are not covered here", g.gu_type,
                           g.up_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    int* ctr = v2_counters(scratch, n, n_expert);
    ck(cudaMemsetAsync(ctr, 0, 2 * sizeof(int), s), "v2 counters");
    const V2Spec sp_gu = pick_v2_gu(g.gu_type, g.up_type);
    const V2Spec sp_d = pick_v2_d(g.d_type);
    const unsigned gr_gu = (unsigned) std::min<int64_t>(tiles * sp_gu.nfb, (int64_t) d.sms * d.occ);
    const unsigned gr_d = (unsigned) std::min<int64_t>(tiles * sp_d.nfb, (int64_t) d.sms * d.occ);
    launch_v2(sp_gu, gr_gu, b, g, tb, xa, src, ha, nullptr, ctr, s);
    launch_v2(sp_d, gr_d, b, g, tb, ha, src, nullptr, dm, ctr + 1, s);
    ck(cudaGetLastError(), "experts_native");
}

}  // namespace basalt::prefill::fused
