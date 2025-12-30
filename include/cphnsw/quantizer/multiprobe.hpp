#pragma once

#include "../core/types.hpp"
#include "cp_encoder.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>

namespace cphnsw {

/**
 * Multiprobe Sequence Generator
 *
 * Generates alternative hash codes ranked by collision probability.
 *
 * From Andoni et al. Section 5:
 * The probability of a collision decreases as we flip to indices with
 * smaller absolute values. The ranking follows:
 *   log(Pr) ∝ -(|x_max| - |x_alt|)²
 *
 * The multiprobe sequence is generated incrementally using a priority queue,
 * exploring alternatives in order of decreasing probability.
 */
template <typename ComponentT, size_t K>
class MultiprobeGenerator {
public:
    /// Probe entry: modified code + log probability
    struct ProbeEntry {
        CPCode<ComponentT, K> code;
        double log_probability;

        bool operator<(const ProbeEntry& other) const {
            // Max-heap: higher probability first
            return log_probability < other.log_probability;
        }
    };

    /**
     * Generate probe sequence from encoded query with sorted indices.
     *
     * @param encoded     Query with full sorted indices per rotation
     * @param max_probes  Maximum number of probes to generate
     * @return            Vector of probes sorted by probability (descending)
     */
    std::vector<ProbeEntry> generate(
        const typename CPEncoder<ComponentT, K>::EncodedWithSortedIndices& encoded,
        size_t max_probes) const {

        std::vector<ProbeEntry> probes;
        probes.reserve(max_probes);

        // Primary probe (original code) has probability 1.0 (log = 0)
        probes.push_back({encoded.code, 0.0});

        if (max_probes == 1) {
            return probes;
        }

        // State for incremental generation
        // modifications[r] = rank of current alternative at rotation r (0 = primary)
        using HeapEntry = std::tuple<CPCode<ComponentT, K>, double, std::vector<size_t>>;
        auto cmp = [](const HeapEntry& a, const HeapEntry& b) {
            return std::get<1>(a) < std::get<1>(b);  // Max-heap by probability
        };
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);

        // Initialize: for each rotation, try flipping to 2nd best
        for (size_t r = 0; r < K; ++r) {
            const auto& sorted = encoded.sorted_indices[r];
            if (sorted.size() < 2) continue;

            CPCode<ComponentT, K> modified = encoded.code;

            Float max_abs = std::get<1>(sorted[0]);
            Float alt_abs = std::get<1>(sorted[1]);
            double delta = max_abs - alt_abs;
            double log_prob = -delta * delta;

            // Encode alternative index with correct sign
            auto [alt_idx, alt_mag, alt_sign] = sorted[1];
            modified.components[r] = CPCode<ComponentT, K>::encode(alt_idx, alt_sign);

            std::vector<size_t> mods(K, 0);
            mods[r] = 1;  // Rotation r modified to rank 1

            heap.push({modified, log_prob, mods});
        }

        // Generate probes greedily
        while (probes.size() < max_probes && !heap.empty()) {
            auto [code, log_prob, mods] = heap.top();
            heap.pop();

            probes.push_back({code, log_prob});

            // Expand: try incrementing each rotation's modification rank
            for (size_t r = 0; r < K; ++r) {
                size_t next_rank = mods[r] + 1;
                const auto& sorted = encoded.sorted_indices[r];

                if (next_rank >= sorted.size()) continue;

                CPCode<ComponentT, K> new_code = code;

                // Compute new log probability
                Float max_abs = std::get<1>(sorted[0]);
                Float new_abs = std::get<1>(sorted[next_rank]);
                Float old_abs = std::get<1>(sorted[mods[r]]);

                double old_delta = max_abs - old_abs;
                double new_delta = max_abs - new_abs;

                // Update log probability (remove old contribution, add new)
                double new_log_prob = log_prob - (new_delta * new_delta - old_delta * old_delta);

                // Encode with correct sign from sorted indices
                auto [idx, mag, is_neg] = sorted[next_rank];
                new_code.components[r] = CPCode<ComponentT, K>::encode(idx, is_neg);

                std::vector<size_t> new_mods = mods;
                new_mods[r] = next_rank;

                heap.push({new_code, new_log_prob, new_mods});
            }
        }

        return probes;
    }

    /**
     * Generate simple probe sequence from CPQuery (without full sorted indices).
     *
     * This is a simplified version that only generates the primary probe
     * plus single-flip alternatives based on stored magnitudes.
     *
     * @param query       Query with primary code and magnitudes
     * @param max_probes  Maximum probes (limited functionality without sorted indices)
     * @return            Vector of probes
     */
    std::vector<ProbeEntry> generate_simple(
        const CPQuery<ComponentT, K>& query,
        size_t max_probes) const {

        std::vector<ProbeEntry> probes;
        probes.reserve(std::min(max_probes, K + 1));

        // Primary probe
        probes.push_back({query.primary_code, 0.0});

        if (max_probes == 1) {
            return probes;
        }

        // For simple version, we can only flip signs (not indices)
        // since we don't have the sorted indices available.
        // Generate probes by flipping each rotation to opposite sign.

        for (size_t r = 0; r < K && probes.size() < max_probes; ++r) {
            CPCode<ComponentT, K> modified = query.primary_code;

            // Flip the sign bit
            ComponentT orig = modified.components[r];
            size_t index = CPCode<ComponentT, K>::decode_index(orig);
            bool was_negative = CPCode<ComponentT, K>::decode_sign_negative(orig);

            modified.components[r] = CPCode<ComponentT, K>::encode(index, !was_negative);

            // Sign flip has lower probability (rough approximation)
            double log_prob = -query.magnitudes[r] * query.magnitudes[r];

            probes.push_back({modified, log_prob});
        }

        // Sort by probability (primary should remain first with log_prob = 0)
        std::stable_sort(probes.begin() + 1, probes.end(),
            [](const ProbeEntry& a, const ProbeEntry& b) {
                return a.log_probability > b.log_probability;
            });

        return probes;
    }
};

}  // namespace cphnsw
