// tests/kernels/cpu/expert_pack_bench.cpp - the CPU expert rows on a pack's own expert blobs.
//
//     build/expert_pack_bench <model.basalt> <layer[,layer]|all> <tokens> <experts> <reps> [threads]
//
// `avx_bench` times these rows on synthetic bytes; this times them on a model's weights.  The blobs come from the
// pack's `experts` section at the offsets its `experts.layout` gives, so they are the bytes the CPU expert pool
// reads in production, and the rows go through the pool's own entry points (`native_gu_rows`, `native_down_rows`,
// and `q2_rows_any`).  `threads` splits a layer's rows across workers: 1 keeps one blob in cache, several stream the
// layer's experts from DRAM.  The outputs are hashed (and checked for token-count bitwise invariance when tokens > 1)
// so two builds of the kernels can be compared bit for bit on real weights.
#include "basalt/core/model_file.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace core = basalt::core;

namespace {

struct Op {
    const char* name;
    uint64_t bytes;               // weight bytes one call reads across its whole row range
    int rows;
    std::function<void(const uint8_t* blob, int r0, int r1)> run;
};

struct PackMeta {
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    float swiglu_limit = 0.f;
    std::vector<float> layer_clamp;
};

PackMeta read_pack_meta(core::ModelFile& mf) {
    PackMeta pm;
    std::string err;
    if (!mf.load_meta(err)) return pm;
    const core::Json& meta = mf.meta();
    if (const core::Json* gj = meta.get("geometry"); gj && gj->is_object()) {
        if (const core::Json* v = gj->get("block_count"); v && v->is_number()) pm.n_layers = v->i64();
        if (const core::Json* v = gj->get("expert_count"); v && v->is_number()) pm.n_expert = v->i64();
        if (const core::Json* v = gj->get("embedding_length"); v && v->is_number()) pm.n_embd = v->i64();
        if (const core::Json* v = gj->get("expert_feed_forward_length"); v && v->is_number()) pm.n_ff = v->i64();
        if (const core::Json* v = gj->get("swiglu_limit"); v && v->is_number()) pm.swiglu_limit = (float) v->num();
        if (const core::Json* v = gj->get("swiglu_clamp_exp")) {
            if (v->is_number()) pm.swiglu_limit = (float) v->num();
            else if (v->is_array())
                for (const core::Json& x : v->items)
                    if (x.is_number()) pm.layer_clamp.push_back((float) x.num());
        }
    }
    if (const core::Json* kv = meta.get("gguf_kv"); kv && kv->is_object()) {
        const core::Json* arch = kv->get("general.architecture");
        if (arch && arch->is_string() && !arch->s.empty()) {
            const std::string pfx = arch->s + ".";
            if (pm.n_layers <= 0)
                if (const core::Json* v = kv->get(pfx + "block_count"); v && v->is_number()) pm.n_layers = v->i64();
            if (pm.n_expert <= 0)
                if (const core::Json* v = kv->get(pfx + "expert_count"); v && v->is_number()) pm.n_expert = v->i64();
            if (pm.n_embd <= 0)
                if (const core::Json* v = kv->get(pfx + "embedding_length"); v && v->is_number()) pm.n_embd = v->i64();
            if (pm.n_ff <= 0)
                if (const core::Json* v = kv->get(pfx + "expert_feed_forward_length"); v && v->is_number())
                    pm.n_ff = v->i64();
            if (pm.swiglu_limit <= 0.f) {
                if (const core::Json* v = kv->get(pfx + "swiglu_limit"); v && v->is_number())
                    pm.swiglu_limit = (float) v->num();
                else if (const core::Json* c = kv->get(pfx + "swiglu_clamp_exp"); c && c->is_number())
                    pm.swiglu_limit = (float) c->num();
            }
        }
    }
    return pm;
}

// `threads` workers each take a slice of the layer's rows and walk every blob.  The slowest thread's wall time is
// the layer's time; one thread keeps a blob in cache, several stream it from DRAM.
double wall_seconds(const Op& op, const std::vector<const uint8_t*>& blobs, int threads, int reps) {
    std::atomic<int> ready{0}, go{0};
    const int per = (op.rows + threads - 1) / threads;
    std::vector<std::thread> th;
    std::vector<double> secs(threads);
    for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
        const int r0 = t * per, r1 = std::min(op.rows, r0 + per);
        ready.fetch_add(1);
        while (!go.load()) {}
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            for (const uint8_t* b : blobs) op.run(b, r0, r1);
        secs[t] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    });
    while (ready.load() < threads) {}
    go.store(1);
    for (auto& x : th) x.join();
    return *std::max_element(secs.begin(), secs.end());
}

uint64_t fnv(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

size_t word_diffs(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, a + i, 4);
        std::memcpy(&ub, b + i, 4);
        if (ua != ub) ++d;
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 6) {
        std::fprintf(stderr, "usage: expert_pack_bench <model.basalt> <layer[,layer]|all> <tokens> <experts> <reps> [threads]\n");
        return 2;
    }
    std::string err;
    core::ModelFile mf;
    if (!mf.open(argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<uint8_t> layout;
    if (!mf.read("experts.layout", layout, err)) { std::fprintf(stderr, "experts.layout: %s\n", err.c_str()); return 1; }
    const core::FileRange experts_range = mf.range("experts");
    if (experts_range.empty()) { std::fprintf(stderr, "no experts section\n"); return 1; }

    const PackMeta pm = read_pack_meta(mf);
    int64_t n_layers = pm.n_layers;
    if (n_layers <= 0) {
        std::stringstream ss(std::string((const char*) layout.data(), layout.size()));
        std::string line;
        while (std::getline(ss, line))
            if (!line.empty() && line[0] != '#' && std::isdigit((unsigned char) line[0])) ++n_layers;
    }

    std::vector<std::pair<int64_t, int64_t>> candidates;
    if (pm.n_embd > 0 && pm.n_ff > 0) candidates.emplace_back(pm.n_embd, pm.n_ff);
    for (auto p : {std::pair<int64_t, int64_t>{2560, 640}, {5120, 2304}, {4096, 1408}})
        if (p.first != pm.n_embd || p.second != pm.n_ff) candidates.push_back(p);

    const std::string layout_text((const char*) layout.data(), layout.size());
    const cpu::ExpertLayout* lay = nullptr;
    int64_t c_embd = 0, c_ff = 0;
    for (auto [ne, nf] : candidates) {
        err.clear();
        if (!cpu::expert_layout_load_text(layout_text, n_layers, pm.n_expert, err, ne, nf, pm.swiglu_limit)) continue;
        if (cpu::expert_layout().n_expert <= 0 || cpu::expert_layout().total != experts_range.bytes) continue;
        lay = &cpu::expert_layout();
        c_embd = ne;
        c_ff = nf;
        break;
    }
    if (!lay) {
        std::fprintf(stderr, "no expert geometry matches the %llu byte experts section (%s)\n",
                     (unsigned long long) experts_range.bytes, err.c_str());
        return 1;
    }

    std::vector<int> layers;
    if (std::string(argv[2]) == "all") {
        for (int64_t l = 0; l < lay->n_layers; ++l) layers.push_back((int) l);
    } else {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int NT = std::atoi(argv[3]);
    const int G = std::atoi(argv[4]), reps = std::atoi(argv[5]);
    const int threads = argc > 6 ? std::atoi(argv[6]) : 1;
    if (NT < 1 || NT > 8 || G < 1 || reps < 1 || threads < 1) { std::fprintf(stderr, "tokens 1..8\n"); return 2; }
    std::printf("expert_pack_bench: %s  n_embd %lld n_ff %lld layers %lld experts/layer %lld\n", cpu::cpu_name().c_str(),
                (long long) c_embd, (long long) c_ff, (long long) lay->n_layers, (long long) lay->n_expert);
    std::printf("  %d tokens, %d experts per layer, %d reps, %d threads\n", NT, G, reps, threads);

    uint64_t total_hash = 1469598103934665603ull;
    size_t tok_diffs = 0;
    for (int l : layers) {
        if (l < 0 || l >= lay->n_layers) { std::printf("layer %d: out of range\n", l); continue; }
        cpu::NativeFmt f = lay->fmt[(size_t) l];
        if ((size_t) l < pm.layer_clamp.size() && pm.layer_clamp[(size_t) l] > 0.f)
            f.swiglu_limit = pm.layer_clamp[(size_t) l];

        // Copy the pack's own blob bytes into RAM ahead of timing so file I/O is not in the kernel loop.
        std::vector<uint8_t> store((size_t) G * f.bytes);
        std::vector<const uint8_t*> blobs(G);
        for (int g = 0; g < G; ++g) {
            const int64_t e = (g * 37 + 5) % lay->n_expert;
            uint8_t* dst = store.data() + (size_t) g * f.bytes;
            if (!core::read_range_at(experts_range, lay->blob_offset(l, e), dst, f.bytes, err)) {
                std::fprintf(stderr, "layer %d expert %lld: %s\n", l, (long long) e, err.c_str());
                return 1;
            }
            blobs[g] = dst;
        }
        const std::string gu_name = f.up_type == f.gu_type
            ? std::string(ggml_type_name((ggml_type) f.gu_type))
            : std::string(ggml_type_name((ggml_type) f.gu_type)) + "+" + ggml_type_name((ggml_type) f.up_type);
        std::printf("layer %2d %-7s/%-6s  blob %zu KiB  rows gu %lld down %lld\n", l, gu_name.c_str(),
                    ggml_type_name((ggml_type) f.d_type), f.bytes >> 10, (long long) f.n_ff, (long long) f.n_embd);

        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<float>> xs(NT, std::vector<float>(f.n_embd));
        std::vector<std::vector<uint8_t>> acts(NT), hqs(NT);
        std::vector<std::vector<float>> ff(NT, std::vector<float>(f.n_ff)), out(NT, std::vector<float>(f.n_embd));
        std::vector<cpu::ActQ> a2_gu(NT), a2_dn(NT);
        const void* pa[8]{};
        const void* ph[8]{};
        const cpu::ActQ* pa2_gu[8]{};
        const cpu::ActQ* pa2_dn[8]{};
        float* fp[8]{};
        float* op_[8]{};
        for (int k = 0; k < NT; ++k) {
            for (auto& v : xs[k]) v = nd(rng);
            acts[k].resize(f.act_bytes);
            hqs[k].resize(f.h_bytes);
            cpu::native_quant_act(f, xs[k].data(), acts[k].data());
            pa[k] = acts[k].data();
            ph[k] = hqs[k].data();
            pa2_gu[k] = &a2_gu[k];
            pa2_dn[k] = &a2_dn[k];
            fp[k] = ff[k].data();
            op_[k] = out[k].data();
        }

        // Same dispatch split as ExpertPool::native_gu_range / native_down_range.
        const bool gu_q2 = cpu::q2_native_kernels(f.gu_type), dn_q2 = cpu::q2_native_kernels(f.d_type);
        if (gu_q2)
            for (int k = 0; k < NT; ++k) cpu::act_quant_any(xs[k].data(), (int) f.n_embd, a2_gu[k]);

        auto run_gu_nt = [&](const uint8_t* b, int nt_call, float* const* ff_dst, int r0, int r1) {
            if (gu_q2) {
                thread_local float gbuf[8][cpu::kNativeMaxFF], ubuf[8][cpu::kNativeMaxFF];
                float* gp[8]{};
                float* up[8]{};
                for (int t = 0; t < nt_call; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                const int nbk = (int) (f.n_embd / 64);
                cpu::q2_rows_any(b, f.gu_row, nbk, pa2_gu, nt_call, gp, r0, r1);
                cpu::q2_rows_any(b + f.up_off, f.gu_row, nbk, pa2_gu, nt_call, up, r0, r1);
                for (int t = 0; t < nt_call; ++t)
                    for (int r = r0; r < r1; ++r) {
                        float g = gbuf[t][r], u = ubuf[t][r];
                        if (f.swiglu_limit > 0.f) {
                            g = std::fmin(g, f.swiglu_limit);
                            u = std::fmax(std::fmin(u, f.swiglu_limit), -f.swiglu_limit);
                        }
                        ff_dst[t][r] = (g / (1.f + std::exp(-g))) * u;
                    }
            } else {
                cpu::native_gu_rows(f, b, pa, nt_call, ff_dst, r0, r1);
            }
        };
        auto run_dn_nt = [&](const uint8_t* b, int nt_call, float* const* out_dst, int r0, int r1) {
            if (dn_q2) cpu::q2_rows_any(b + f.down_off, f.d_row, (int) (f.n_ff / 64), pa2_dn, nt_call, out_dst, r0, r1);
            else cpu::native_down_rows(f, b, ph, nt_call, out_dst, r0, r1);
        };

        Op gu{"gate/up", f.down_off, (int) f.n_ff, [&](const uint8_t* b, int r0, int r1) {
            run_gu_nt(b, NT, fp, r0, r1);
        }};
        Op dn{"down", (uint64_t) f.d_row * (uint64_t) f.n_embd, (int) f.n_embd, [&](const uint8_t* b, int r0, int r1) {
            run_dn_nt(b, NT, op_, r0, r1);
        }};
        auto make_h = [&] {
            for (int k = 0; k < NT; ++k)
                if (dn_q2) cpu::act_quant_any(ff[k].data(), (int) f.n_ff, a2_dn[k]);
                else cpu::native_quant_h(f, ff[k].data(), hqs[k].data());
        };

        gu.run(blobs[0], 0, (int) f.n_ff);
        make_h();
        for (Op* op : {&gu, &dn}) {
            for (int i = 0; i < 2; ++i) wall_seconds(*op, blobs, threads, 1);
            double best = 1e30;
            for (int round = 0; round < 3; ++round) best = std::min(best, wall_seconds(*op, blobs, threads, reps));
            std::printf("   %-8s %8.1f us/call  %6.2f GB/s\n", op->name, 1e6 * best / reps,
                        (double) op->bytes * G * reps / best / 1e9);
        }
        std::vector<float> ff1((size_t) f.n_ff), out1((size_t) f.n_embd);
        float* fp1[1] = {ff1.data()};
        float* op1[1] = {out1.data()};
        for (const uint8_t* b : blobs) {
            gu.run(b, 0, (int) f.n_ff);
            make_h();
            dn.run(b, 0, (int) f.n_embd);
            for (int k = 0; k < NT; ++k) { total_hash ^= fnv(ff[k].data(), f.n_ff); total_hash *= 1099511628211ull; }
            for (int k = 0; k < NT; ++k) { total_hash ^= fnv(out[k].data(), f.n_embd); total_hash *= 1099511628211ull; }
            if (NT > 1) {
                run_gu_nt(b, 1, fp1, 0, (int) f.n_ff);
                run_dn_nt(b, 1, op1, 0, (int) f.n_embd);
                tok_diffs += word_diffs(ff[0].data(), ff1.data(), (size_t) f.n_ff);
                tok_diffs += word_diffs(out[0].data(), out1.data(), (size_t) f.n_embd);
            }
        }
    }
    if (NT > 1)
        std::printf("expert_pack_bench: outputs hash %016llx  (token-0 vs 1-token diffs: %zu)\n",
                    (unsigned long long) total_hash, tok_diffs);
    else
        std::printf("expert_pack_bench: outputs hash %016llx\n", (unsigned long long) total_hash);
    return tok_diffs == 0 ? 0 : 1;
}
