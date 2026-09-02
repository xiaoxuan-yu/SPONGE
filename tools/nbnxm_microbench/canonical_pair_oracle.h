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

struct CanonicalPairOccurrence
{
    CanonicalPair pair;
    size_t sci_index = 0;
    size_t packed_index = 0;
    int split = 0;
    int jm = 0;
    int i_local = 0;
    int cluster_i = -1;
    int cluster_j = -1;
    int sci_shift_id = 13;
    int pair_shift_id = 13;
    int exclusion_index = 0;
    unsigned int imask = 0;
    uint64_t exclusion_hash = 0;
};

struct CanonicalPairSourceSummary
{
    size_t sci_index = 0;
    size_t packed_index = 0;
    int split = 0;
    int jm = 0;
    int i_local = 0;
    int cluster_i = -1;
    int cluster_j = -1;
    int sci_shift_id = 13;
    int pair_shift_id = 13;
    int exclusion_index = 0;
    unsigned int imask = 0;
    uint64_t exclusion_hash = 0;
    size_t accepted_pairs = 0;
    size_t duplicate_pairs = 0;
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
    std::vector<CanonicalPairOccurrence> first_duplicate_occurrences;
    std::vector<CanonicalPairSourceSummary> duplicate_source_summaries;
    std::vector<CanonicalPair> first_missing;
    std::vector<CanonicalPair> first_extra;
    std::string failure_reason;
};

CanonicalPairOracleResult CompareCanonicalPairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    size_t example_limit = 8);

}  // namespace nbnxm_microbench
