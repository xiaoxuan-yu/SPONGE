#ifndef SPONGE_CLUSTERED_NEIGHBOR_CPU_BUILDER_H
#define SPONGE_CLUSTERED_NEIGHBOR_CPU_BUILDER_H

#ifdef USE_CPU

#include <vector>

#include "../contract/traversal.cuh"

namespace clustered_neighbor_cpu_builder
{

struct BuildInput
{
    int local_atom_numbers = 0;
    int cluster_size = kClusteredClusterSize;
    int candidate_sci_numbers = 0;
    bool dense_shift_partitioned_candidates = false;
    float cutoff = 0.0f;
    LTMatrix3 cell = {};

    std::vector<int> permutation;
    std::vector<int> cluster_offsets;
    std::vector<int> super_cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<VECTOR> cluster_centers;
    std::vector<VECTOR> cluster_extents;
    std::vector<int> leaf_cluster_starts;
    std::vector<int> leaf_cluster_ends;
    std::vector<int> cluster_to_supercluster;
    std::vector<int> sci_supercluster_ids;
    std::vector<int> candidate_shift_ids;
    std::vector<int> candidate_leaf_offsets;
    std::vector<int> candidate_leaf_ids;
    std::vector<int> excluded_list_start;
    std::vector<int> excluded_list;
    std::vector<int> excluded_numbers;
};

struct BuildOutput
{
    std::vector<CLUSTERED_GMXPACKED_SCI> gmxpacked_scis;
    std::vector<CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked;
    std::vector<CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_exclusions;
    int gmxpacked_split_exclusion_numbers = 0;
};

void BuildPayload(const BuildInput& input, BuildOutput* output);

}  // namespace clustered_neighbor_cpu_builder

#endif

#endif
