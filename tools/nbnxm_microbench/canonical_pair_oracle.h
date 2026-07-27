#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nbnxm_microbench_snapshot.h"

namespace nbnxm_microbench
{

struct CanonicalPair
{
    int global_i = -1;
    int global_j = -1;
    int shift_id = 13;

    bool operator==(const CanonicalPair&) const = default;
    bool operator<(const CanonicalPair& other) const
    {
        if (global_i != other.global_i)
        {
            return global_i < other.global_i;
        }
        if (global_j != other.global_j)
        {
            return global_j < other.global_j;
        }
        return shift_id < other.shift_id;
    }
};

struct CanonicalPairOracleResult
{
    bool metadata_ready = false;
    bool matched = false;
    size_t payload_pair_count = 0;
    size_t oracle_pair_count = 0;
    size_t duplicate_payload_pairs = 0;
    size_t missing_pairs = 0;
    size_t extra_pairs = 0;
    std::vector<CanonicalPair> first_duplicates;
    std::vector<CanonicalPair> first_missing;
    std::vector<CanonicalPair> first_extra;
    std::string failure_reason;
};

CanonicalPairOracleResult CompareCanonicalPairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    size_t example_limit = 8);

}  // namespace nbnxm_microbench
