// tests/models/qwen4exp/kernels/kv_stream_paged_parity.cpp - Paged HostKvPool / HostKvView parity & microbench.
//
// Verifies every paged INT8 host KV kernel touchpoint against:
//   (a) the contiguous KvHostPools Basalt kernels (bitwise-equal output: max abs = 0, max rel = 0), and
//   (b) a plain double-precision CPU reference (dequantize INT8 * FP16 scale, FP64 softmax & weighted value sum),
// plus batch/window width and row-position independence, multi-slab (>32,768 pages) permuted page tables, COW prefix
// sharing + divergence, prefill staging (`kv_stage_from_host`), and VRAM <-> HostKvPool scatter/warm transitions.
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/models/qwen4exp/decode/host_kv_pool.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_fused.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_seg.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace k = basalt::kernels;
namespace c = basalt::core;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t i = 0; i < n_kv; ++i) ids.push_back((int32_t) i);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    const int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t i = n_bid * 4; i < n_kv; ++i) ids.push_back((int32_t) i);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// Plain CPU double-precision QSA attention over selected cells from a paged HostKvPool + HostKvTable.
void cpu_attn_fp64(const float* q, const c::HostKvPool& pool, const c::HostKvTable& tab, const int32_t* ids,
                   int width, const k::QsaShapes& s, float* out) {
    const int NH = (int) s.n_head, NKV = (int) s.n_head_kv, D = (int) s.head_dim, G = NH / NKV;
    const double scale = 1.0 / std::sqrt((double) D);
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / G;
        const float* qh = q + (size_t) h * D;
        std::vector<double> scores((size_t) width, -1e300);
        double smax = -1e300;
        for (int j = 0; j < width; ++j) {
            const int cell = ids[j], blk = cell / 4, sub = cell % 4;
            const k::HostKvPage* hp = pool.page_ptr(tab.h_pages[(size_t) blk]);
            const int lrow = kvh * 4 + sub;
            const int8_t* kc = hp->k_q + lrow * D;
            const uint16_t* ks = hp->k_scale + lrow * 4;
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
                const double kf = (double) k::f32_from_f16(ks[d / 64]) * (double) kc[d];
                // Match kv_gather_q8's fp16 dequantized K value
                const double k16 = (double) k::f32_from_f16(k::f16_from_f32((float) kf));
                dot += (double) qh[d] * k16;
            }
            scores[(size_t) j] = dot * scale;
            smax = std::max(smax, scores[(size_t) j]);
        }
        double denom = 0.0;
        for (int j = 0; j < width; ++j) {
            scores[(size_t) j] = std::exp(scores[(size_t) j] - smax);
            denom += scores[(size_t) j];
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int j = 0; j < width; ++j) {
                const int cell = ids[j], blk = cell / 4, sub = cell % 4;
                const k::HostKvPage* hp = pool.page_ptr(tab.h_pages[(size_t) blk]);
                const int lrow = kvh * 4 + sub;
                const double vf = (double) k::f32_from_f16(hp->v_scale[lrow * 4 + d / 64]) *
                                  (double) hp->v_q[lrow * D + d];
                const double v16 = (double) k::f32_from_f16(k::f16_from_f32((float) vf));
                acc += (scores[(size_t) j] / denom) * v16;
            }
            out[(size_t) h * D + d] = (float) acc;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), NQ = 8;
    // 34,000 blocks = 136,000 cells -> spans TWO 32,768-page slabs in HostKvPool!
    const int64_t n_blocks = 34000, N = n_blocks * 4, n_slots = 5120;
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    c::HostKvPool pool(true);
    std::string err;

    // Pre-allocate and permute physical pages across 2 slabs so logical blocks map non-contiguously across slabs
    c::HostKvTable tab0, tab1;
    tab0.h_pages.assign((size_t) n_blocks, -1);
    tab1.h_pages.assign((size_t) n_blocks, -1);
    tab0.d_pages = dalloc<int32_t>((size_t) n_blocks);
    tab1.d_pages = dalloc<int32_t>((size_t) n_blocks);
    if (!pool.ensure_writable(tab0, 0, n_blocks, err)) {
        std::fprintf(stderr, "ensure_writable failed: %s\n", err.c_str());
        return 1;
    }
    std::shuffle(tab0.h_pages.begin(), tab0.h_pages.end(), rng);
    ck(cudaMemcpy(tab0.d_pages, tab0.h_pages.data(), (size_t) n_blocks * 4, cudaMemcpyHostToDevice), "upload permuted");

    auto make_vram_slots = [&](int64_t pages) {
        k::KvHostPools p;
        const size_t rows = (size_t) pages * H * 4;
        p.k_q = dalloc<int8_t>(rows * D);
        p.v_q = dalloc<int8_t>(rows * D);
        p.k_scale = dalloc<uint16_t>(rows * 4);
        p.v_scale = dalloc<uint16_t>(rows * 4);
        return p;
    };
    auto make_host_contig = [&](int64_t pages) {
        k::KvHostPools p;
        const size_t rows = (size_t) pages * H * 4;
        p.k_q = halloc<int8_t>(rows * D);
        p.v_q = halloc<int8_t>(rows * D);
        p.k_scale = halloc<uint16_t>(rows * 4);
        p.v_scale = halloc<uint16_t>(rows * 4);
        return p;
    };
    auto make_map = [&]() {
        k::KvStreamMap m;
        m.page_table = dalloc<int32_t>((size_t) n_blocks);
        m.slot_block = dalloc<int32_t>((size_t) n_slots);
        m.slot_stamp = dalloc<int32_t>((size_t) n_slots);
        m.slot_ref = dalloc<int32_t>((size_t) n_slots);
        m.miss_block = dalloc<int32_t>((size_t) n_slots);
        m.miss_slot = dalloc<int32_t>((size_t) n_slots);
        m.ctl = dalloc<int32_t>(k::kKvCtlInts);
        m.n_blocks = n_blocks;
        m.n_slots = n_slots;
        k::kv_stream_reset(m, nullptr);
        return m;
    };
    auto attn_pools = [](const k::KvHostPools& p, int32_t* pt) {
        k::QsaAttnPools a;
        a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale; a.v_scale = p.v_scale; a.page_table = pt;
        return a;
    };

    k::KvHostPools contig0 = make_host_contig(n_blocks);
    k::KvHostPools slots_contig = make_vram_slots(n_slots), slots_paged = make_vram_slots(n_slots);
    k::KvStreamMap map_contig = make_map(), map_paged = make_map();
    k::KvHostPools host_paged0{};
    host_paged0.paged = {pool.device_slabs(0), tab0.d_pages};

    // 1. Prefill chunk append (`basalt::prefill::kv_append`) for [0, 134000) across the 32,768-page slab boundary
    const int64_t pf_len = 134000, chunk = 4096;
    float* d_K = dalloc<float>((size_t) chunk * H * D);
    float* d_V = dalloc<float>((size_t) chunk * H * D);
    std::vector<float> hK((size_t) chunk * H * D), hV((size_t) chunk * H * D);
    for (int64_t p0 = 0; p0 < pf_len; p0 += chunk) {
        const int64_t T = std::min(chunk, pf_len - p0);
        for (size_t i = 0; i < (size_t) (T * H * D); ++i) {
            hK[i] = nd(rng) * ((i % 29 == 0) ? 12.0f : 1.0f);
            hV[i] = nd(rng);
        }
        ck(cudaMemcpy(d_K, hK.data(), (size_t) (T * H * D) * 4, cudaMemcpyHostToDevice), "pf K");
        ck(cudaMemcpy(d_V, hV.data(), (size_t) (T * H * D) * 4, cudaMemcpyHostToDevice), "pf V");
        basalt::prefill::kv_append(d_K, d_V, T, p0, map_contig.page_table, 4, nullptr, nullptr,
                                   slots_contig.k_q, slots_contig.v_q, slots_contig.k_scale, slots_contig.v_scale,
                                   nullptr, &contig0, nullptr);
        basalt::prefill::kv_append(d_K, d_V, T, p0, map_paged.page_table, 4, nullptr, nullptr,
                                   slots_paged.k_q, slots_paged.v_q, slots_paged.k_scale, slots_paged.v_scale,
                                   nullptr, &host_paged0, nullptr);
    }

    // 2. Decode appends (`kv_append_q8_step` and `qsa_dec_kv_post`) for [134000, 136000)
    float* d_kcur = dalloc<float>((size_t) NQ * H * D);
    float* d_kcur2 = dalloc<float>((size_t) NQ * H * D);
    float* d_vcur = dalloc<float>((size_t) NQ * H * D);
    float* d_knorm = dalloc<float>((size_t) D);
    int32_t* d_posk = dalloc<int32_t>((size_t) NQ * NH);
    int32_t* d_steps = dalloc<int32_t>((size_t) NQ * k::kStepCount);
    {
        std::vector<float> kn((size_t) D, 1.0f);
        ck(cudaMemcpy(d_knorm, kn.data(), kn.size() * 4, cudaMemcpyHostToDevice), "knorm");
    }
    const k::RopeScaling rscale{};
    for (int64_t pos = pf_len; pos < N;) {
        const int T = (pos % 16 == 0) ? 1 : std::min<int>(4, (int) (N - pos));
        std::vector<float> hk((size_t) T * H * D), hv((size_t) T * H * D);
        std::vector<int32_t> hp((size_t) T * NH), hs((size_t) T * 4);
        for (float& x : hk) x = nd(rng);
        for (float& x : hv) x = nd(rng);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) hp[(size_t) t * NH + h] = (int32_t) (pos + t);
            hs[t * 4 + 0] = (int32_t) (pos + t);
            hs[t * 4 + 1] = (int32_t) (pos + t + 1);
            hs[t * 4 + 2] = (int32_t) ((pos + t + 1) / 4);
            hs[t * 4 + 3] = 0;
        }
        ck(cudaMemcpy(d_kcur, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k1");
        ck(cudaMemcpy(d_kcur2, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k2");
        ck(cudaMemcpy(d_vcur, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(d_posk, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "pos");
        ck(cudaMemcpy(d_steps, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "steps");
        if (T == 1) {
            k::kv_append_q8_step(slots_contig.k_q, slots_contig.v_q, slots_contig.k_scale, slots_contig.v_scale,
                                 map_contig.page_table, d_steps, d_kcur, d_vcur, s, nullptr, &contig0);
            k::kv_append_q8_step(slots_paged.k_q, slots_paged.v_q, slots_paged.k_scale, slots_paged.v_scale,
                                 map_paged.page_table, d_steps, d_kcur2, d_vcur, s, nullptr, &host_paged0);
        } else {
            k::QsaDecKvPost kp1{}, kp2{};
            kp1.kcur = d_kcur; kp1.vcur = d_vcur; kp1.pos_k = d_posk; kp1.pos_stride = (int) NH;
            kp1.k_norm = d_knorm; kp1.steps = d_steps; kp1.part = 1;
            kp1.k_q = slots_contig.k_q; kp1.v_q = slots_contig.v_q; kp1.k_scale = slots_contig.k_scale;
            kp1.v_scale = slots_contig.v_scale; kp1.page_table = map_contig.page_table; kp1.host = contig0;
            kp2 = kp1;
            kp2.kcur = d_kcur2;
            kp2.k_q = slots_paged.k_q; kp2.v_q = slots_paged.v_q; kp2.k_scale = slots_paged.k_scale;
            kp2.v_scale = slots_paged.v_scale; kp2.page_table = map_paged.page_table; kp2.host = host_paged0;
            k::qsa_dec_kv_post(kp1, T, s, rscale, nullptr);
            k::qsa_dec_kv_post(kp2, T, s, rscale, nullptr);
        }
        pos += T;
    }
    ck(cudaDeviceSynchronize(), "sync appends");

    // 3. Attention parity: paged vs contiguous (BITWISE), paged vs FP64 CPU reference, and window-width independence
    int32_t* d_ids = dalloc<int32_t>((size_t) NQ * cap);
    float* d_q = dalloc<float>((size_t) NQ * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* d_scr = dalloc<float>((size_t) NQ * scr);
    float* d_out_contig = dalloc<float>((size_t) NQ * NH * D);
    float* d_out_paged = dalloc<float>((size_t) NQ * NH * D);
    float* d_out_single = dalloc<float>(NH * D);

    double max_abs_contig = 0.0, max_rel_contig = 0.0;
    double max_abs_fp64 = 0.0, max_rel_fp64 = 0.0;
    int bitwise_mismatches = 0, width_mismatches = 0;

    for (int trial = 0; trial < 16; ++trial) {
        const int n_q = 1 + (trial % NQ);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        std::vector<float> hq((size_t) NQ * NH * D);
        for (float& x : hq) x = nd(rng);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = N - n_q + t, n_kv = p + 1, width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv;
            hst[t * 4 + 2] = (int32_t) (n_kv / 4); hst[t * 4 + 3] = (int32_t) width;
        }
        ck(cudaMemcpy(d_ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");

        k::kv_stream_resolve(map_contig, attn_pools(slots_contig, map_contig.page_table), contig0, k::kKvInt8,
                             d_ids, d_steps, n_q, cap, s, nullptr);
        k::qsa_dec_attn(d_q, attn_pools(slots_contig, map_contig.page_table), d_ids, d_steps, cap, s, d_scr,
                        d_out_contig, n_q, nullptr);

        k::kv_stream_resolve(map_paged, attn_pools(slots_paged, map_paged.page_table), host_paged0, k::kKvInt8,
                             d_ids, d_steps, n_q, cap, s, nullptr);
        k::qsa_dec_attn(d_q, attn_pools(slots_paged, map_paged.page_table), d_ids, d_steps, cap, s, d_scr,
                        d_out_paged, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "resolve+attn");

        std::vector<float> oc((size_t) n_q * NH * D), op((size_t) n_q * NH * D);
        ck(cudaMemcpy(oc.data(), d_out_contig, oc.size() * 4, cudaMemcpyDeviceToHost), "oc");
        ck(cudaMemcpy(op.data(), d_out_paged, op.size() * 4, cudaMemcpyDeviceToHost), "op");
        if (std::memcmp(oc.data(), op.data(), oc.size() * 4) != 0) ++bitwise_mismatches;
        for (size_t i = 0; i < oc.size(); ++i) {
            const double diff = std::abs((double) oc[i] - (double) op[i]);
            max_abs_contig = std::max(max_abs_contig, diff);
            if (std::abs(oc[i]) > 1e-12) max_rel_contig = std::max(max_rel_contig, diff / std::abs((double) oc[i]));
        }

        // Check batch/window width & position independence: query `n_q - 1` run alone (width 1) matches row `n_q - 1`
        const int r = n_q - 1;
        k::kv_stream_resolve(map_paged, attn_pools(slots_paged, map_paged.page_table), host_paged0, k::kKvInt8,
                             d_ids + (size_t) r * cap, d_steps + r * 4, 1, cap, s, nullptr);
        k::qsa_dec_attn(d_q + (size_t) r * NH * D, attn_pools(slots_paged, map_paged.page_table),
                        d_ids + (size_t) r * cap, d_steps + r * 4, cap, s, d_scr, d_out_single, 1, nullptr);
        std::vector<float> os((size_t) NH * D);
        ck(cudaMemcpy(os.data(), d_out_single, os.size() * 4, cudaMemcpyDeviceToHost), "os");
        if (std::memcmp(os.data(), op.data() + (size_t) r * NH * D, os.size() * 4) != 0) ++width_mismatches;

        // Compare row 0 against plain double-precision CPU reference on the first 4 trials
        if (trial < 4) {
            std::vector<float> ref64((size_t) NH * D);
            cpu_attn_fp64(hq.data(), pool, tab0, hids.data(), hst[3], s, ref64.data());
            for (size_t i = 0; i < ref64.size(); ++i) {
                const double diff = std::abs((double) op[i] - (double) ref64[i]);
                max_abs_fp64 = std::max(max_abs_fp64, diff);
                if (std::abs(ref64[i]) > 1e-4) max_rel_fp64 = std::max(max_rel_fp64, diff / std::abs((double) ref64[i]));
            }
        }
    }

    // 4. COW clone_range + segmented batch `qsa_dec_kv_post_seg` + `kv_stream_resolve_seg` parity
    if (!pool.clone_range(tab1, tab0, 132000, err) || !pool.ensure_writable(tab1, 33000, n_blocks, err)) {
        std::fprintf(stderr, "COW clone/ensure failed: %s\n", err.c_str());
        return 1;
    }
    k::KvHostPools host_paged1{};
    host_paged1.paged = {pool.device_slabs(0), tab1.d_pages};
    k::KvHostPools slots_paged1 = make_vram_slots(n_slots);
    k::KvStreamMap map_paged1 = make_map();

    k::QsaSegSlot h_seg_slots[2] = {};
    h_seg_slots[0].k_q = slots_paged.k_q; h_seg_slots[0].v_q = slots_paged.v_q;
    h_seg_slots[0].k_scale = slots_paged.k_scale; h_seg_slots[0].v_scale = slots_paged.v_scale;
    h_seg_slots[0].page_table = map_paged.page_table; h_seg_slots[0].host = host_paged0; h_seg_slots[0].map = map_paged;
    h_seg_slots[1] = h_seg_slots[0];
    h_seg_slots[1].k_q = slots_paged1.k_q; h_seg_slots[1].v_q = slots_paged1.v_q;
    h_seg_slots[1].k_scale = slots_paged1.k_scale; h_seg_slots[1].v_scale = slots_paged1.v_scale;
    h_seg_slots[1].page_table = map_paged1.page_table; h_seg_slots[1].host = host_paged1; h_seg_slots[1].map = map_paged1;

    k::QsaSegSlot* d_seg_slots = dalloc<k::QsaSegSlot>(2);
    k::QsaSeg* d_rows = dalloc<k::QsaSeg>(k::kQsaSegMaxRows);
    k::QsaSeg* d_segs = dalloc<k::QsaSeg>(k::kQsaSegMaxRows);
    ck(cudaMemcpy(d_seg_slots, h_seg_slots, sizeof(h_seg_slots), cudaMemcpyHostToDevice), "seg slots");
    std::vector<k::QsaSeg> hrows(k::kQsaSegMaxRows, {-1, 0, 0, 0}), hsegs(k::kQsaSegMaxRows, {-1, 0, 0, 0});
    hsegs[0] = {1, 0, 2, 0};
    hsegs[1] = {0, 2, 2, 0};
    hrows[0] = hrows[1] = hsegs[0];
    hrows[2] = hrows[3] = hsegs[1];
    ck(cudaMemcpy(d_rows, hrows.data(), hrows.size() * sizeof(k::QsaSeg), cudaMemcpyHostToDevice), "d_rows");
    ck(cudaMemcpy(d_segs, hsegs.data(), hsegs.size() * sizeof(k::QsaSeg), cudaMemcpyHostToDevice), "d_segs");

    // Append 2 divergent tokens to slot 1 at [132000, 132002) and 2 tokens to slot 0 at [135996, 135998)
    {
        std::vector<float> hk((size_t) 4 * H * D), hv((size_t) 4 * H * D), hq((size_t) 4 * NH * D);
        for (float& x : hk) x = nd(rng) * 3.0f;
        for (float& x : hv) x = nd(rng) * 3.0f;
        for (float& x : hq) x = nd(rng);
        std::vector<int32_t> hp((size_t) 4 * NH), hs(16), hids((size_t) 4 * cap, 0);
        const int64_t positions[4] = {132000, 132001, 135996, 135997};
        for (int r = 0; r < 4; ++r) {
            for (int h = 0; h < NH; ++h) hp[(size_t) r * NH + h] = (int32_t) positions[r];
            hs[r * 4 + 0] = (int32_t) positions[r];
            hs[r * 4 + 1] = (int32_t) (positions[r] + 1);
            hs[r * 4 + 2] = (int32_t) ((positions[r] + 1) / 4);
            hs[r * 4 + 3] = (int32_t) cap;
            const std::vector<int32_t> sel = selection(positions[r] + 1, cap, rng);
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) r * cap);
        }
        ck(cudaMemcpy(d_kcur, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "seg k");
        ck(cudaMemcpy(d_vcur, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "seg v");
        ck(cudaMemcpy(d_posk, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "seg pos");
        ck(cudaMemcpy(d_steps, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "seg steps");
        ck(cudaMemcpy(d_ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "seg ids");
        ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "seg q");

        k::QsaSegTable seg_tab{d_seg_slots, d_rows, d_segs, 4, 2};
        k::QsaDecKvPost kp{};
        kp.kcur = d_kcur; kp.vcur = d_vcur; kp.pos_k = d_posk; kp.pos_stride = (int) NH;
        kp.k_norm = d_knorm; kp.steps = d_steps; kp.part = 1;
        k::qsa_dec_kv_post_seg(kp, seg_tab, 4, s, rscale, nullptr);
        k::kv_stream_resolve_seg(seg_tab, d_ids, d_steps, cap, s, nullptr);
        k::qsa_dec_attn_seg(d_q, seg_tab, true, d_ids, d_steps, cap, s, d_scr, d_out_paged, 4, nullptr);

        // Compare each segment's rows against solo resolve + solo attention on its own slot
        k::kv_stream_resolve(map_paged1, attn_pools(slots_paged1, map_paged1.page_table), host_paged1, k::kKvInt8,
                             d_ids, d_steps, 2, cap, s, nullptr);
        k::qsa_dec_attn(d_q, attn_pools(slots_paged1, map_paged1.page_table), d_ids, d_steps, cap, s, d_scr,
                        d_out_contig, 2, nullptr);
        k::kv_stream_resolve(map_paged, attn_pools(slots_paged, map_paged.page_table), host_paged0, k::kKvInt8,
                             d_ids + 2 * cap, d_steps + 8, 2, cap, s, nullptr);
        k::qsa_dec_attn(d_q + 2 * NH * D, attn_pools(slots_paged, map_paged.page_table), d_ids + 2 * cap,
                        d_steps + 8, cap, s, d_scr, d_out_contig + 2 * NH * D, 2, nullptr);
        ck(cudaDeviceSynchronize(), "seg vs solo");
        std::vector<float> o_seg((size_t) 4 * NH * D), o_solo((size_t) 4 * NH * D);
        ck(cudaMemcpy(o_seg.data(), d_out_paged, o_seg.size() * 4, cudaMemcpyDeviceToHost), "o_seg");
        ck(cudaMemcpy(o_solo.data(), d_out_contig, o_solo.size() * 4, cudaMemcpyDeviceToHost), "o_solo");
        if (std::memcmp(o_seg.data(), o_solo.data(), o_seg.size() * 4) != 0) ++width_mismatches;
    }

    // 5. `kv_stage_from_host` + `kv_scatter_to_host_paged` + `kv_stream_warm_recent` round-trip byte equality
    int stage_mismatches = 0;
    {
        const int64_t test_blocks = 6000;
        k::KvHostPools stage_contig = make_vram_slots(test_blocks), stage_paged = make_vram_slots(test_blocks);
        int32_t* ident = dalloc<int32_t>((size_t) test_blocks);
        k::kv_ring_table(ident, test_blocks, test_blocks, nullptr);
        k::kv_stage_from_host(attn_pools(stage_contig, ident), contig0, k::kKvInt8, test_blocks, s, nullptr);
        k::kv_stage_from_host(attn_pools(stage_paged, ident), host_paged0, k::kKvInt8, test_blocks, s, nullptr);

        c::HostKvTable tab_rt;
        tab_rt.h_pages.assign((size_t) test_blocks, -1);
        tab_rt.d_pages = dalloc<int32_t>((size_t) test_blocks);
        pool.ensure_writable(tab_rt, 0, test_blocks, err);
        k::HostKvView view_rt{pool.device_slabs(0), tab_rt.d_pages};
        k::kv_scatter_to_host_paged(attn_pools(stage_paged, ident), view_rt, test_blocks, s, nullptr);
        ck(cudaDeviceSynchronize(), "stage+scatter");

        const size_t codes_n = (size_t) test_blocks * 2048;
        std::vector<uint8_t> sc(codes_n), sp(codes_n);
        ck(cudaMemcpy(sc.data(), stage_contig.k_q, codes_n, cudaMemcpyDeviceToHost), "sc");
        ck(cudaMemcpy(sp.data(), stage_paged.k_q, codes_n, cudaMemcpyDeviceToHost), "sp");
        if (sc != sp) ++stage_mismatches;
        for (int64_t b = 0; b < test_blocks; ++b) {
            if (std::memcmp(pool.page_ptr(tab_rt.h_pages[(size_t) b]), pool.page_ptr(tab0.h_pages[(size_t) b]),
                            sizeof(k::HostKvPage)) != 0) {
                ++stage_mismatches;
                break;
            }
        }
        // Verify kv_stream_warm_recent places blocks [test_blocks - n_slots, test_blocks) in slots [0, n_slots)
        k::KvStreamMap wm = make_map();
        k::kv_stream_warm_recent(wm, attn_pools(stage_paged, wm.page_table), test_blocks, s, nullptr);
        ck(cudaDeviceSynchronize(), "warm recent");
        std::vector<int32_t> pt((size_t) n_blocks), sb((size_t) n_slots);
        ck(cudaMemcpy(pt.data(), wm.page_table, pt.size() * 4, cudaMemcpyDeviceToHost), "wm pt");
        ck(cudaMemcpy(sb.data(), wm.slot_block, sb.size() * 4, cudaMemcpyDeviceToHost), "wm sb");
        const int64_t b0 = test_blocks - n_slots;
        for (int64_t sl = 0; sl < n_slots; ++sl)
            if (sb[(size_t) sl] != b0 + sl || pt[(size_t) (b0 + sl)] != sl) { ++stage_mismatches; break; }
        pool.release_table(tab_rt);
    }

    std::printf("kv_stream_paged_parity:\n"
                "  vs contiguous KvHostPools: bitwise_mismatches=%d max_abs=%.3g max_rel=%.3g\n"
                "  vs CPU FP64 reference:     max_abs=%.3g max_rel=%.3g\n"
                "  window/seg independence:   mismatches=%d | stage/scatter/warm mismatches=%d\n",
                bitwise_mismatches, max_abs_contig, max_rel_contig, max_abs_fp64, max_rel_fp64,
                width_mismatches, stage_mismatches);

    if (bench) {
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "e0");
        ck(cudaEventCreate(&e1), "e1");
        constexpr int kIters = 50;
        double us_contig = 0, us_paged = 0;
        uint64_t misses = 0;
        for (int it = 0; it < kIters; ++it) {
            std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
            for (int t = 0; t < NQ; ++t) {
                const int64_t p = N - NQ + t, n_kv = p + 1;
                const std::vector<int32_t> sel = selection(n_kv, cap, rng);
                std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
                hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv;
                hst[t * 4 + 2] = (int32_t) (n_kv / 4); hst[t * 4 + 3] = (int32_t) cap;
            }
            ck(cudaMemcpy(d_ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "b ids");
            ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "b steps");
            k::kv_stream_reset(map_contig, nullptr);
            k::kv_stream_reset(map_paged, nullptr);
            ck(cudaDeviceSynchronize(), "b reset");

            const uint64_t m0 = k::kv_stream_counters(map_paged).misses;
            ck(cudaEventRecord(e0, nullptr), "r0");
            k::kv_stream_resolve(map_contig, attn_pools(slots_contig, map_contig.page_table), contig0, k::kKvInt8,
                                 d_ids, d_steps, NQ, cap, s, nullptr);
            ck(cudaEventRecord(e1, nullptr), "r1");
            ck(cudaEventSynchronize(e1), "s1");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "el1");
            us_contig += ms * 1000.0 / kIters;

            ck(cudaEventRecord(e0, nullptr), "r2");
            k::kv_stream_resolve(map_paged, attn_pools(slots_paged, map_paged.page_table), host_paged0, k::kKvInt8,
                                 d_ids, d_steps, NQ, cap, s, nullptr);
            ck(cudaEventRecord(e1, nullptr), "r3");
            ck(cudaEventSynchronize(e1), "s2");
            ck(cudaEventElapsedTime(&ms, e0, e1), "el2");
            us_paged += ms * 1000.0 / kIters;
            misses += k::kv_stream_counters(map_paged).misses - m0;
        }
        const double bytes = (double) misses / kIters * 4224.0;
        std::printf("  bench cold 8-query resolve (%.2f MB/call): contiguous %.1f us (%.1f GB/s) | paged %.1f us (%.1f GB/s)\n",
                    bytes / 1e6, us_contig, bytes / 1e9 / (us_contig * 1e-6),
                    us_paged, bytes / 1e9 / (us_paged * 1e-6));
    }

    pool.release_table(tab0);
    pool.release_table(tab1);
    const bool ok = bitwise_mismatches == 0 && width_mismatches == 0 && stage_mismatches == 0 && max_abs_fp64 < 5e-3;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
