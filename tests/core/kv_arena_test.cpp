// The elastic KV pools (kv_arena.hpp).  Mapping backs whole pages of every array, zeroed; unmapping gives the VRAM
// back and keeps the cells below the cut exactly as they were; the addresses never move.  Needs a CUDA device with
// virtual memory management (exits 77 without one).
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

// every byte of the first `bytes` of `p` equals `v`
bool all(const uint8_t* p, uint64_t bytes, uint8_t v) {
    std::vector<uint8_t> h((size_t) bytes);
    if (cudaMemcpy(h.data(), p, (size_t) bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (const uint8_t x : h)
        if (x != v) return false;
    return true;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    // INT8 geometry of one QSA layer: per page of 64 cells, 4 KV heads x 128 codes (K, V) and 2 scales each (K, V)
    constexpr int64_t kPage = 64, kPages = 4096;
    const std::vector<uint64_t> page_bytes = {kPage * 4 * 128, kPage * 4 * 128, kPage * 4 * 2 * 2, kPage * 4 * 2 * 2};
    basalt::core::KvArena a;
    std::string err;
    if (!a.open(page_bytes, kPages, kPage, err)) {
        if (err.find("virtual memory management") != std::string::npos) {
            std::printf("%s: skipped\n", err.c_str());
            return 77;
        }
        return fail("open", err);
    }
    if (a.cells() != 0 || a.mapped_bytes() != 0 || a.max_cells() != kPages * kPage) return fail("open sizes");
    std::vector<uint8_t*> base;
    for (size_t i = 0; i < page_bytes.size(); ++i) base.push_back(a.array(i));
    // back 10000 cells: whole pages, and whole granules of each array
    if (!a.map(10000, err)) return fail("map", err);
    if (a.cells() < 10000 || a.cells() % kPage != 0) return fail("map cells");
    if (a.grow_bytes(10000) != 0 || a.mapped_bytes() != a.bytes_for(10000)) return fail("map bytes");
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t b = (uint64_t) a.cells() / kPage * page_bytes[i];
        if (!all(a.array(i), b, 0)) return fail("fresh pages are zero");
        if (cudaMemset(a.array(i), (int) (i + 1), (size_t) b) != cudaSuccess) return fail("write");
    }
    // back 200000 cells: the old cells kept, the new ones zero, the same addresses
    const int64_t c0 = a.cells();
    const size_t before = free_vram();
    const uint64_t add = a.grow_bytes(200000);
    if (add == 0 || !a.map(200000, err) || a.cells() < 200000) return fail("map more", err);
    if (before - free_vram() + (4u << 20) < add) return fail("VRAM taken");
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        if (a.array(i) != base[i]) return fail("address moved");
        const uint64_t b0 = (uint64_t) c0 / kPage * page_bytes[i], b1 = (uint64_t) 200000 / kPage * page_bytes[i];
        if (!all(a.array(i), b0, (uint8_t) (i + 1))) return fail("old cells kept");
        if (!all(a.array(i) + b0, b1 - b0 - page_bytes[i], 0)) return fail("new cells zero");
    }
    // give back down to 10000: the VRAM back, the first cells as they were
    const size_t mid = free_vram();
    if (!a.unmap(10000, err) || a.cells() != c0) return fail("unmap", err);
    if (free_vram() - mid + (4u << 20) < add) return fail("VRAM not given back");
    for (size_t i = 0; i < page_bytes.size(); ++i)
        if (!all(a.array(i), (uint64_t) c0 / kPage * page_bytes[i], (uint8_t) (i + 1))) return fail("kept after unmap");
    // to nothing, then the whole context
    if (!a.unmap(0, err) || a.cells() != 0 || a.mapped_bytes() != 0) return fail("unmap all", err);
    if (!a.map(a.max_cells(), err) || a.cells() != a.max_cells()) return fail("map all", err);
    a.close();
    if (a.valid()) return fail("close");

    // a row array with a tail (the indexer's pooled keys: a row per page and two spare rows) and a table filled from
    // the host (the rope tables) beside a cache array
    {
        constexpr uint64_t kRow = 512, kTab = 128;
        std::vector<uint8_t> table((size_t) (kPages * kTab));
        for (size_t i = 0; i < table.size(); ++i) table[i] = (uint8_t) (i * 7 + 3);
        basalt::core::KvArena b;
        if (!b.open({kPage * 64, kRow, kPage * kTab / 64}, kPages, kPage, err, {0, 2 * kRow, 0})) return fail("open tail", err);
        b.set_contents(2, table);
        if (!b.map(1, err) || b.cells() < kPage) return fail("map one page", err);
        // the first page and the spare rows past it are backed
        if (!all(b.array(1), 3 * kRow, 0)) return fail("rows with a tail");
        const int64_t want = 10000;
        if (!b.map(want, err) || b.cells() < want) return fail("map tail", err);
        const uint64_t pages = (uint64_t) b.cells() / kPage;
        if (!all(b.array(1), (pages + 2) * kRow, 0)) return fail("tail rows backed");
        std::vector<uint8_t> got((size_t) (pages * kPage * kTab / 64));
        if (cudaMemcpy(got.data(), b.array(2), got.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("read table");
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != table[i]) return fail("table filled from the host");
        // given back and mapped again, the table is filled again; the cells count never runs ahead of the tail
        if (!b.unmap(0, err) || b.cells() != 0 || !b.map(want, err)) return fail("remap", err);
        if (cudaMemcpy(got.data(), b.array(2), got.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("read table");
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != table[i]) return fail("table refilled");
        if (!b.map(b.max_cells(), err) || b.cells() != b.max_cells() || !all(b.array(1), (kPages + 2) * kRow, 0))
            return fail("map all with a tail", err);
    }
    // on a grid, one deep map is given back to the first grid line at or above the cut, the cells below kept
    {
        constexpr int64_t kGrid = 128 * kPage;
        basalt::core::KvArena d;
        if (!d.open(page_bytes, kPages, kPage, err)) return fail("open grid", err);
        d.set_extent_cells(kGrid);
        if (!d.map(kGrid, err) || !d.map(8 * kGrid, err) || d.cells() < 8 * kGrid) return fail("map grid", err);
        if (cudaMemset(d.array(0), 0x5a, (size_t) (3 * kGrid / kPage * page_bytes[0])) != cudaSuccess) return fail("fill grid");
        if (!d.unmap(2 * kGrid + 1, err) || d.cells() != 3 * kGrid || d.mapped_bytes() != d.bytes_for(3 * kGrid))
            return fail("unmap to the grid", err);
        if (!all(d.array(0), 3 * kGrid / kPage * page_bytes[0], 0x5a)) return fail("kept below the grid line");
        basalt::core::KvArena one;
        if (!one.open(page_bytes, kPages, kPage, err) || !one.map(kGrid, err) || !one.map(8 * kGrid, err) ||
            !one.unmap(2 * kGrid + 1, err) || one.mapped_bytes() != one.bytes_for(8 * kGrid))
            return fail("one mapping per call stays", err);
    }
    // split QSA elastic arenas: unmapping K/V down to 20,480 cells keeps idx_pooled and cos/sin mapped at 260,000
    // while returning ~2.8 GiB of VRAM across the 12 QSA layers, and pre-carves KvStreamMap at fixed addresses
    {
        basalt::core::ModelGeometry g{};
        constexpr int64_t kMaxCells = 262144, kFloor = 4096, kStream = 20480, kProbeCell = 260000;
        basalt::core::qsa_set_kv_int8(true);
        basalt::core::qsa_set_kv_elastic(kFloor);
        const uint64_t sbytes = basalt::core::session_bytes(g, kMaxCells, 10);
        void* sbuf = nullptr;
        if (cudaMalloc(&sbuf, sbytes) != cudaSuccess) return fail("session alloc");
        basalt::core::SessionState s;
        if (basalt::core::session_init(g, kMaxCells, 10, sbuf, s) == 0) {
            cudaFree(sbuf);
            return fail("session init");
        }
        for (int64_t j = 0; j < s.qsa_alloc; ++j) {
            const basalt::core::QsaState& q = s.qsa_states[j];
            if (q.kv_arena == nullptr || q.idx_arena == nullptr) return fail("split arenas missing");
            if (q.map.ctl == nullptr || q.map.n_slots != kStream / 4) return fail("pre-carved stream map");
        }
        if (!basalt::core::session_kv_map(s, kMaxCells, err) || basalt::core::session_kv_cells(s) != kMaxCells)
            return fail("session_kv_map 256k", err);
        const int64_t probe_row = kProbeCell / 4;
        std::vector<float> marker((size_t) g.idx_key_dim, 42.5f);
        for (int64_t j = 0; j < s.qsa_alloc; ++j) {
            if (cudaMemcpy(s.qsa_states[j].idx_pooled + (size_t) probe_row * g.idx_key_dim, marker.data(),
                           marker.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("write idx_pooled at 260k");
        }
        const size_t free_full = free_vram();
        const uint64_t kv_before = basalt::core::session_kv_only_mapped_bytes(s);
        if (!basalt::core::session_kv_only_unmap(s, kStream, err)) return fail("session_kv_only_unmap", err);
        const uint64_t kv_after = basalt::core::session_kv_only_mapped_bytes(s);
        const size_t free_shrunk = free_vram();
        const uint64_t freed_arena = kv_before - kv_after;
        if (freed_arena < (2800ull << 20) || free_shrunk <= free_full || free_shrunk - free_full < (2750ull << 20))
            return fail("2.8 GiB VRAM returned");
        // verify idx_pooled and primary RoPE tables at cell 260,000 remain backed and readable
        std::vector<float> got_idx((size_t) g.idx_key_dim, 0.0f), got_cos(32, 0.0f), got_sin(32, 0.0f);
        for (int64_t j = 0; j < s.qsa_alloc; ++j) {
            if (cudaMemcpy(got_idx.data(), s.qsa_states[j].idx_pooled + (size_t) probe_row * g.idx_key_dim,
                           got_idx.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("read idx_pooled after kv_only_unmap");
            if (got_idx != marker) return fail("idx_pooled preserved after kv_only_unmap");
        }
        if (cudaMemcpy(got_cos.data(), s.qsa_states[0].cos_tab + (size_t) kProbeCell * 32, 32 * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(got_sin.data(), s.qsa_states[0].sin_tab + (size_t) kProbeCell * 32, 32 * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("read cos/sin at 260k after kv_only_unmap");
        float norm_sq = got_cos[0] * got_cos[0] + got_sin[0] * got_sin[0];
        if (norm_sq < 0.999f || norm_sq > 1.001f) return fail("cos^2 + sin^2 == 1 at 260k");
        basalt::core::session_release(s);
        delete[] s.qsa_states;
        cudaFree(sbuf);
        basalt::core::qsa_set_kv_elastic(0);
        basalt::core::qsa_set_kv_int8(false);
    }
    std::puts("kv arena: whole pages backed and zeroed on demand, given back past the cut, addresses fixed, mappings "
              "on a grid, split K/V and indexer arenas verified");
    return 0;
}
