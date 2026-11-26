#pragma once

#include <cstdint>
#include <limits>

namespace lfs::denoise {

class HistoryLifecycle {
private:
    bool valid_{};
    bool pending_{};
    std::uint32_t write_index_{};
    std::uint64_t expected_source_generation_{};
    std::uint64_t pending_source_generation_{};

public:
    void reset() noexcept {
        valid_ = false;
        pending_ = false;
        write_index_ = 0u;
        expected_source_generation_ = 0u;
        pending_source_generation_ = 0u;
    }

    void begin_frame() noexcept {
        pending_ = false;
        pending_source_generation_ = 0u;
    }

    void mark_pending(std::uint64_t bound_source_generation) noexcept {
        pending_ = true;
        pending_source_generation_ =
            bound_source_generation ==
                    std::numeric_limits<std::uint64_t>::max()
                ? 1u
                : bound_source_generation + 1u;
    }

    void cancel_pending() noexcept {
        pending_ = false;
        pending_source_generation_ = 0u;
    }

    void commit_success() noexcept {
        if (!pending_) return;
        write_index_ ^= 1u;
        valid_ = true;
        expected_source_generation_ = pending_source_generation_;
        pending_ = false;
        pending_source_generation_ = 0u;
    }

    [[nodiscard]] bool source_generation_valid(
        bool source_history_valid,
        std::uint64_t source_generation) const noexcept {
        return valid_
            ? source_history_valid && expected_source_generation_ != 0u &&
                source_generation == expected_source_generation_
            : !source_history_valid && source_generation == 0u &&
                expected_source_generation_ == 0u;
    }

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    [[nodiscard]] std::uint32_t write_index() const noexcept {
        return write_index_;
    }
    [[nodiscard]] std::uint32_t read_index() const noexcept {
        return write_index_ ^ 1u;
    }
    [[nodiscard]] std::uint64_t expected_source_generation() const noexcept {
        return expected_source_generation_;
    }
};

} // namespace lfs::denoise
