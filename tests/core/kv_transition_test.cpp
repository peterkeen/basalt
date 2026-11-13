// Dynamic QSA tier changes and prefix copies preserve every authoritative byte.
#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::core;
namespace {
int checks = 0;
void check(bool ok, const char* what, const std::string& err = {}) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s: %s\n", what, err.c_str()); std::exit(1); }
}
void cu(cudaError_t status) { check(status == cudaSuccess, "CUDA", cudaGetErrorString(status)); }
struct Fixture {
    ModelGeometry g;
    QsaState st;
    void* buf = nullptr;
    static constexpr int64_t cells = 65536;
    Fixture() {
        const uint64_t bytes = qsa_state_bytes(g, cells);
        cu(cudaMalloc(&buf, bytes));
        check(qsa_state_init(g, cells, buf, st) > 0, "state init");
        std::string err;
        check(st.kv_arena->map(cells, err) && st.idx_arena->map(cells, err), "map", err);
    }
    ~Fixture() { qsa_state_release(st); cudaFree(buf); }
    void fill(int salt) {
        const std::array<void*, 5> ptr = {st.k_q, st.v_q, st.k_scale, st.v_scale, st.idx_pooled};
        const std::array<size_t, 5> sizes = {cells * 512, cells * 512, cells * 16, cells * 16,
                                           (cells / 4 + 2) * 128 * sizeof(float)};
        for (size_t a = 0; a < ptr.size(); ++a) {
            std::vector<uint8_t> data(sizes[a]);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (salt + a * 31 + i * 7 + i / 257);
            cu(cudaMemcpy(ptr[a], data.data(), data.size(), cudaMemcpyHostToDevice));
        }
    }
};
void verify(const ConversationKv& image, Fixture& f, int64_t live) {
    std::string err;
    uint64_t hash = 0;
    check(conversation_kv_verify(image, f.st, f.g, live, true, hash, err), "bitwise KV and indexer parity", err);
}
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    qsa_set_kv_int8(true);
    qsa_set_kv_resident(0);
    qsa_set_kv_elastic(4096);
    {
        Fixture a, b;
        std::string err;
        auto* const k = a.st.k_q;
        auto* const table = a.st.page_table;
        auto* const ctl = a.st.map.ctl;
        // Partial pages, both sides of the staging floor, and overlapping warm ranges.
        for (const int64_t live : {0ll, 3ll, 20480ll, 20481ll, 24579ll, 65533ll}) {
            check(a.st.kv_arena->map(Fixture::cells, err), "remap", err);
            a.fill(13);
            ConversationKv image;
            check(conversation_kv_save(image, a.st, a.g, live, true, err), "save reference", err);
            const uint64_t idx_bytes = a.st.idx_arena->mapped_bytes();
            check(qsa_state_demote_to_stream(a.st, a.g, live, err), "demote", err);
            check(a.st.kv_mode == 1 && a.st.n_slots == 5120, "stream mode");
            check(a.st.k_q == k && a.st.page_table == table && a.st.map.ctl == ctl, "fixed addresses");
            check(a.st.idx_arena->mapped_bytes() == idx_bytes, "indexer stays mapped");
            verify(image, a, live);
            const int64_t pages = (live + 3) / 4;
            const int64_t warm = std::min<int64_t>(pages, a.st.n_slots), first = pages - warm;
            std::vector<int32_t> map((size_t) a.st.n_pages);
            cu(cudaMemcpy(map.data(), a.st.page_table, map.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int64_t p = 0; p < pages; ++p)
                check(map[(size_t) p] == (p < first ? -1 : p - first), "warm page table");
            // Every warmed byte agrees with its authoritative host page, including overlapping ranges.
            const std::array<const void*, 4> ptr = {a.st.k_q, a.st.v_q, a.st.k_scale, a.st.v_scale};
            const std::array<size_t, 4> page_bytes = {2048, 2048, 64, 64};
            for (size_t arr = 0; arr < ptr.size() && warm > 0; ++arr) {
                std::vector<uint8_t> got((size_t) warm * page_bytes[arr]);
                cu(cudaMemcpy(got.data(), ptr[arr], got.size(), cudaMemcpyDeviceToHost));
                for (int64_t slot = 0; slot < warm; ++slot) {
                    const auto* hp = a.st.host_pool->page_ptr(a.st.host_tab.h_pages[(size_t) (first + slot)]);
                    const std::array<const void*, 4> expected = {hp->k_q, hp->v_q, hp->k_scale, hp->v_scale};
                    check(std::memcmp(got.data() + (size_t) slot * page_bytes[arr], expected[arr], page_bytes[arr]) == 0,
                          "warm K/V and scales");
                }
            }
            check(qsa_state_promote_to_vram(a.st, a.g, live, live + 1, err), "promote", err);
            check(a.st.kv_mode == 0 && a.st.n_slots == a.st.n_pages && !a.st.host.present(), "resident mode");
            verify(image, a, live);
            cu(cudaMemcpy(map.data(), a.st.page_table, map.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int64_t p = 0; p < a.st.n_pages; ++p) check(map[(size_t) p] == p, "identity page table");
            check(qsa_host_kv_pool().pinned_bytes() == 0, "all slabs freed after promotion");
        }
        constexpr int64_t live = 24579;
        check(a.st.kv_arena->map(Fixture::cells, err), "map copy source", err);
        a.fill(29);
        ConversationKv image;
        check(conversation_kv_save(image, a.st, a.g, live, true, err), "save copy reference", err);
        check(qsa_state_demote_to_stream(b.st, b.g, 0, err), "empty destination demote", err);
        check(conversation_kv_copy(a.st, b.st, a.g, live, true, err), "resident to streamed", err);
        verify(image, b, live);
        check(qsa_state_demote_to_stream(a.st, a.g, live, err), "source demote", err);
        const int64_t pages_before = qsa_host_kv_pool().live_pages();
        check(conversation_kv_copy(b.st, a.st, a.g, live, true, err), "streamed clone", err);
        verify(image, a, live);
        check(qsa_host_kv_pool().live_pages() < pages_before, "clone releases destination pages");
        const int64_t last = live / 4;
        const int32_t shared = a.st.host_tab.h_pages[(size_t) last];
        check(shared == b.st.host_tab.h_pages[(size_t) last] && a.st.host_pool->refcount(shared) == 2, "shared prefix");
        check(qsa_host_kv_ensure(a.st, live, live + 1, err), "partial-page COW", err);
        check(a.st.host_tab.h_pages[(size_t) last] != shared, "COW splits boundary page");
        verify(image, a, live);
        verify(image, b, live);
        check(qsa_state_promote_to_vram(a.st, a.g, live, Fixture::cells, err), "promote shared source", err);
        a.fill(177);
        check(conversation_kv_copy(b.st, a.st, a.g, live, true, err), "streamed to resident", err);
        verify(image, a, live);
        check(qsa_state_reset_to_vram(b.st, 4096, err), "discard streamed session", err);
        check(qsa_host_kv_pool().pinned_bytes() == 0, "discard frees all slabs");
    }
    qsa_set_kv_elastic(0);
    qsa_set_kv_int8(false);
    std::printf("kv transitions: %d checks passed, bitwise round trips and cross-tier copies verified\n", checks);
    return 0;
}
