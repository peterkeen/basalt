// src/models/qwen4exp/prefill/moe_fused_v2.hpp - the v2 native fused prompt expert kernel variants, one TU each.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "basalt/models/qwen4exp/prefill/moe_fused_iq.hpp"

namespace basalt::prefill::fused {

constexpr int kV2Threads = 256;

struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};

struct V2Spec {
    const void* fn;
    int smem;
    int nfb;
};

V2Spec v2_spec_iq2xxs();
V2Spec v2_spec_iq2xs();
V2Spec v2_spec_iq2s();
V2Spec v2_spec_iq3xxs();
V2Spec v2_spec_iq3s();
V2Spec v2_spec_iq4xs();
V2Spec v2_spec_mix_iq2s_iq2xxs();
V2Spec v2_spec_q2_0();
V2Spec v2_spec_iq4nl();

}  // namespace basalt::prefill::fused
