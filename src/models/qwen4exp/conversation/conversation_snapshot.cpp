#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "conversation_checked.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <limits>

namespace basalt::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = basalt::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode != 0 || st.kv_q4 || st.kv_int8)) {
        error = "conversation snapshot: hybrid K8V4 requires an identity layout and distinct format flags";
        return false;
    }
    const auto s = basalt::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V
    // (KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) basalt::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid) return {st.k_q, st.v_q4, st.k_scale, nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode == 1 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
        if (st.host.paged.present() &&
            (st.host_pool == nullptr || (int64_t) st.host_tab.h_pages.size() < l.cells / l.page_size)) {
            error = "conversation snapshot: invalid paged host KV table";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

// An elastic state backs only a prefix of its cells; a copy past it would fault inside the driver.
bool backed(const QsaState& st, const Layout& l, std::string& error) {
    if (st.kv_mode == 0 && st.kv_arena && st.kv_arena->cells() < l.cells) {
        error = "conversation snapshot: " + std::to_string((long long) l.cells) + " cells to copy, " +
                std::to_string((long long) st.kv_arena->cells()) + " backed";
        return false;
    }
    if (l.pooled > 0 && st.idx_arena && st.idx_arena->cells() < l.cells) {
        error = "conversation snapshot: " + std::to_string((long long) l.cells) + " indexer cells to copy, " +
                std::to_string((long long) st.idx_arena->cells()) + " backed";
        return false;
    }
    return true;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes, bool borrow_host) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error) || !backed(st, l, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const bool ring_only = st.kv_mode == 2 && !st.host.present();
    const bool paged_host = st.kv_mode == 1 && st.host.paged.present() && st.host_pool != nullptr;
    const auto src = pools(st, ring_only);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const int64_t n1 = l.cells / l.page_size;
    const int64_t ring_b0 = ring_only ? std::max<int64_t>(0, n1 - st.n_slots) : 0;
    auto page_slice = [](const basalt::kernels::HostKvPage* hp, size_t arr_idx) -> const uint8_t* {
        if (hp == nullptr) return nullptr;
        switch (arr_idx) {
            case 0: return reinterpret_cast<const uint8_t*>(hp->k_q);
            case 1: return reinterpret_cast<const uint8_t*>(hp->v_q);
            case 2: return reinterpret_cast<const uint8_t*>(hp->k_scale);
            case 3: return reinterpret_cast<const uint8_t*>(hp->v_scale);
            default: return nullptr;
        }
    };
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / basalt::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        if (borrow_host && paged_host && i < 4 && n1 > 0) {
            std::vector<const uint8_t*> pages((size_t) n1);
            for (int64_t page = 0; page < n1; ++page) {
                const int32_t phys = st.host_tab.h_pages[(size_t) page];
                pages[(size_t) page] = phys >= 0 ? page_slice(st.host_pool->page_ptr(phys), i) : nullptr;
                if (!pages[(size_t) page]) { error = "conversation snapshot: unallocated spill page"; return false; }
            }
            if (!dst[i]->borrow_pages(std::move(pages), sizes[i] / (size_t) n1)) return false;
            continue;
        }
        dst[i]->resize_for_overwrite(sizes[i]);   // every new byte is copied in below
        if (paged_host && i < 4 && sizes[i] > 0 && n1 > 0) {
            // Direct CPU read from `HostKvPool` pages: zero GPU D2H traffic for K/V.
            const size_t page_bytes = sizes[i] / (size_t) n1;
            const int64_t p_start = (int64_t) (keep / page_bytes);
            for (int64_t page = p_start; page < n1; ++page) {
                const int32_t phys = st.host_tab.h_pages[(size_t) page];
                const uint8_t* ps = phys >= 0 ? page_slice(st.host_pool->page_ptr(phys), i) : nullptr;
                const size_t off = (size_t) page * page_bytes;
                const size_t from = std::max(keep, off);
                const size_t span = page_bytes - (from - off);
                dst[i]->visit(from, span, [&](uint8_t* p, size_t n, size_t at) {
                    if (ps != nullptr) std::memcpy(p, ps + (from - off) + (at - from), n);
                    else std::memset(p, 0, n);
                    return true;
                });
            }
        } else if (ring_only && i < 4 && sizes[i] > 0 && n1 > 0) {
            const size_t page_bytes = sizes[i] / (size_t) n1;
            const size_t zend = (size_t) ring_b0 * page_bytes;
            if (keep < zend)
                dst[i]->visit(keep, zend - keep, [](uint8_t* p, size_t n, size_t) { std::memset(p, 0, n); return true; });
            const int64_t p_start = std::max<int64_t>(ring_b0, (int64_t) (keep / page_bytes));
            for (int64_t page = p_start; page < n1;) {
                const int64_t sl = page % st.n_slots;
                const int64_t run = std::min<int64_t>(n1 - page, st.n_slots - sl);
                const size_t off = (size_t) page * page_bytes;
                const size_t from = std::max(keep, off);
                const size_t span = (size_t) run * page_bytes - (from - off);
                const uint8_t* ring_src = static_cast<const uint8_t*>(src[i]) + (size_t) sl * page_bytes + (from - off);
                if (!dst[i]->visit(from, span, [&](uint8_t* p, size_t n, size_t at) {
                        return transfer(p, ring_src + (at - from), n, error);
                    })) return false;
                page += run;
            }
        } else if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const bool paged_host = st.kv_mode == 1 && st.host.paged.present() && st.host_pool != nullptr;
    const auto dst = pools(st, st.kv_mode == 2 && !st.host.present());
    for (size_t i = 0; i < src.size(); ++i) {
        if (paged_host && i < 4) continue;
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    }
    return true;
}

namespace {
// After the authoritative pools are written: the resident copies refill from them before any attention reads.
bool settle_resident(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error) {
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) basalt::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && st.host.present() && upto > 0) {
        auto shapes = basalt::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        basalt::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}
}  // namespace

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !backed(st, l, error)) return false;
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const bool ring_only = st.kv_mode == 2 && !st.host.present();
    const bool paged_host = st.kv_mode == 1 && st.host.paged.present() && st.host_pool != nullptr;
    const auto dst = pools(st, ring_only);
    const int64_t n1 = l.cells / l.page_size;
    const int64_t ring_b0 = ring_only ? std::max<int64_t>(0, n1 - st.n_slots) : 0;
    if (paged_host) {
        HostKvTable& tab = const_cast<HostKvTable&>(st.host_tab);
        st.host_pool->release_range(tab, n1, /*keep_spare=*/true);
        if (n1 > 0 && !st.host_pool->ensure_writable(tab, 0, n1, error, nullptr, /*zero_fresh=*/false)) return false;
    }
    auto page_slice_mut = [](basalt::kernels::HostKvPage* hp, size_t arr_idx) -> uint8_t* {
        if (hp == nullptr) return nullptr;
        switch (arr_idx) {
            case 0: return reinterpret_cast<uint8_t*>(hp->k_q);
            case 1: return reinterpret_cast<uint8_t*>(hp->v_q);
            case 2: return reinterpret_cast<uint8_t*>(hp->k_scale);
            case 3: return reinterpret_cast<uint8_t*>(hp->v_scale);
            default: return nullptr;
        }
    };
    for (size_t i = 0; i < src.size(); ++i) {
        if (paged_host && i < 4) {
            if (src[i]->empty() || n1 <= 0) continue;
            const size_t page_bytes = src[i]->size() / (size_t) n1;
            for (int64_t page = 0; page < n1; ++page) {
                uint8_t* pd = page_slice_mut(st.host_pool->page_ptr(st.host_tab.h_pages[(size_t) page]), i);
                if (pd == nullptr) { error = "conversation snapshot: unallocated host KV page"; return false; }
                const size_t off = (size_t) page * page_bytes;
                src[i]->visit(off, page_bytes, [&](const uint8_t* p, size_t n, size_t at) {
                    std::memcpy(pd + (at - off), p, n);
                    return true;
                });
            }
            continue;
        }
        if (ring_only && i < 4) {
            if (src[i]->empty() || n1 <= 0) continue;
            const size_t page_bytes = src[i]->size() / (size_t) n1;
            for (int64_t page = ring_b0; page < n1;) {
                const int64_t sl = page % st.n_slots;
                const int64_t run = std::min<int64_t>(n1 - page, st.n_slots - sl);
                const size_t off = (size_t) page * page_bytes;
                uint8_t* ring_dst = static_cast<uint8_t*>(dst[i]) + (size_t) sl * page_bytes;
                if (!src[i]->visit(off, (size_t) run * page_bytes, [&](const uint8_t* p, size_t n, size_t at) {
                        return transfer(ring_dst + (at - off), p, n, error);
                    })) return false;
                page += run;
            }
            continue;
        }
        if (!src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    }
    return settle_resident(st, g, upto, error);
}

bool conversation_kv_copy(QsaState& from, QsaState& to, const ModelGeometry& g, int64_t upto, bool index,
                          std::string& error) {
    Layout a{}, b{};
    if (!layout(from, g, upto, index, a, error) || !valid(from, a, upto, error) || !backed(from, a, error) ||
        !layout(to, g, upto, index, b, error) || !valid(to, b, upto, error) || !backed(to, b, error))
        return false;
    if (a.format != b.format || a.cells != b.cells || a.pooled_rows != b.pooled_rows || a.page_size != b.page_size ||
        a.data != b.data || a.scales != b.scales || a.pooled != b.pooled || a.value_data != b.value_data ||
        a.value_scales != b.value_scales) {
        error = "conversation snapshot: the two sessions' K/V layouts differ";
        return false;
    }
    const bool from_ring_only = from.kv_mode == 2 && !from.host.present();
    const bool to_ring_only = to.kv_mode == 2 && !to.host.present();
    if (from_ring_only != to_ring_only || (from_ring_only && from.n_slots != to.n_slots)) {
        error = "conversation snapshot: incompatible draft ring layouts";
        return false;
    }
    const int64_t n1 = a.cells / a.page_size;
    const bool from_paged = from.kv_mode == 1 && from.host.paged.present() && from.host_pool != nullptr;
    const bool to_paged = to.kv_mode == 1 && to.host.paged.present() && to.host_pool != nullptr;
    if (from_paged || to_paged) {
        auto shapes = basalt::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv;
        shapes.head_dim = g.head_dim;
        if (from_paged && to_paged && from.host_pool == to.host_pool) {
            if (!to.host_pool->clone_range(to.host_tab, from.host_tab, upto, error)) return false;
        } else if (!from_paged && from.kv_mode == 0 && to_paged) {
            to.host_pool->release_range(to.host_tab, n1, /*keep_spare=*/true);
            if (n1 > 0 && !to.host_pool->ensure_writable(to.host_tab, 0, n1, error, nullptr, /*zero_fresh=*/false)) return false;
            if (n1 > 0) basalt::kernels::kv_scatter_to_host_paged(qsa_attn_pools(from), to.host.paged, n1, shapes, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                error = std::string("conversation snapshot scatter: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        } else if (from_paged && !to_paged && to.kv_mode == 0) {
            if (n1 > 0) basalt::kernels::kv_stage_from_host(qsa_attn_pools(to), from.host, basalt::kernels::kKvInt8, n1, shapes, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                error = std::string("conversation snapshot gather: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        } else {
            // Fallback across distinct HostKvPools or mixed streaming modes.
            ConversationKv tmp;
            return conversation_kv_save(tmp, from, g, upto, index, error) &&
                   conversation_kv_restore(tmp, to, g, upto, index, error);
        }
        if (a.pooled > 0 && !transfer(to.idx_pooled, from.idx_pooled, a.pooled, error)) return false;
        return settle_resident(to, g, upto, error);
    }
    const auto src = pools(from, from_ring_only);
    const auto dst = pools(to, to_ring_only);
    const std::array<size_t,5> sizes = {a.data, a.value_data, a.scales, a.value_scales, a.pooled};
    const int64_t b0 = from_ring_only ? std::max<int64_t>(0, n1 - from.n_slots) : 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
        if (from_ring_only && i < 4) {
            if (!sizes[i] || n1 <= 0) continue;
            const size_t page_bytes = sizes[i] / (size_t) n1;
            for (int64_t page = b0; page < n1;) {
                const int64_t sl = page % from.n_slots;
                const int64_t run = std::min<int64_t>(n1 - page, from.n_slots - sl);
                if (!transfer(static_cast<uint8_t*>(dst[i]) + (size_t) sl * page_bytes,
                              static_cast<const uint8_t*>(src[i]) + (size_t) sl * page_bytes,
                              (size_t) run * page_bytes, error))
                    return false;
                page += run;
            }
            continue;
        }
        if (!transfer(dst[i], src[i], sizes[i], error)) return false;
    }
    return settle_resident(to, g, upto, error);
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const bool ring_only = st.kv_mode == 2 && !st.host.present();
    const bool paged_host = st.kv_mode == 1 && st.host.paged.present() && st.host_pool != nullptr;
    const auto authoritative = pools(st, ring_only);
    for (size_t i = 0; i < saved.size(); ++i) {
        if (paged_host && i < 4) {
            if (saved[i]->empty()) continue;
            const int64_t end = image.cells / image.page_size;
            const size_t page_bytes = saved[i]->size() / (size_t) end;
            uint8_t zeros[2048] = {};
            for (int64_t page = 0; page < end; ++page) {
                const int32_t phys = st.host_tab.h_pages[(size_t) page];
                const basalt::kernels::HostKvPage* hp = phys >= 0 ? st.host_pool->page_ptr(phys) : nullptr;
                const uint8_t* actual = hp == nullptr ? zeros
                    : i == 0 ? reinterpret_cast<const uint8_t*>(hp->k_q)
                    : i == 1 ? reinterpret_cast<const uint8_t*>(hp->v_q)
                    : i == 2 ? reinterpret_cast<const uint8_t*>(hp->k_scale)
                             : reinterpret_cast<const uint8_t*>(hp->v_scale);
                bool ok = true;
                saved[i]->visit((size_t) page * page_bytes, page_bytes, [&](const uint8_t* p, size_t n, size_t at) {
                    const size_t off = at - (size_t) page * page_bytes;
                    if (std::memcmp(actual + off, p, n) != 0) { ok = false; return false; }
                    for (size_t k = 0; k < n; ++k) { hash ^= actual[off + k]; hash *= 1099511628211ull; }
                    return true;
                });
                if (!ok) { error = "conversation snapshot: restored K/V bytes differ"; return false; }
            }
            continue;
        }
        if (ring_only && i < 4) {
            if (saved[i]->empty()) continue;
            const int64_t end = image.cells / image.page_size;
            const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
            const size_t page_bytes = saved[i]->size() / (size_t) end;
            if (begin > 0) {
                bool zeros_ok = true;
                saved[i]->visit(0, (size_t) begin * page_bytes, [&](const uint8_t* p, size_t n, size_t) {
                    for (size_t k = 0; k < n; ++k) {
                        if (p[k] != 0) { zeros_ok = false; return false; }
                        hash ^= 0u;
                        hash *= 1099511628211ull;
                    }
                    return true;
                });
                if (!zeros_ok) { error = "conversation snapshot: hostless draft ring prefix is nonzero"; return false; }
            }
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit((size_t) page * page_bytes, page_bytes, [&](const uint8_t* p, size_t n, size_t at) {
                        const size_t offset = (size_t) (page % st.n_slots) * page_bytes + at - (size_t) page * page_bytes;
                        return compare(static_cast<const uint8_t*>(authoritative[i]) + offset, p, n, true);
                    })) return false;
            continue;
        }
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    }
    if (st.kv_mode == 2 && st.host.present() && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

} // namespace basalt::core
