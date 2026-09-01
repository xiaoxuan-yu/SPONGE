static CLUSTERED_GMXPACKED_EXCLUSION Make_Empty_Gmxpacked_No_Exclusion()
{
    CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
    for (unsigned int& pair_word : exclusion.pair)
    {
        pair_word = 0xffffffffu;
    }
    return exclusion;
}

static __host__ __device__ __forceinline__ bool Clustered_Split_Has_Atoms(
    unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static __host__ __device__ __forceinline__ void
Fill_Gmxpacked_Record_Stream_Source_Pair_Words(
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* source,
    const int cluster_i_start, const int cluster_j, const int shift_id,
    const int split_id, const unsigned int valid_mask_j,
    const unsigned int local_mask_j, const unsigned int record_imask,
    const unsigned int* shared_i_local_masks,
    const unsigned long long* exclusion_masks)
{
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        source->pair_exclusion_words[pair_idx] = 0u;
    }

#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int source_imask_bit =
            1u << static_cast<unsigned int>(i_local);
        if ((record_imask & source_imask_bit) == 0u)
        {
            continue;
        }

        const int cluster_i = cluster_i_start + i_local;
        const unsigned int local_mask_i = shared_i_local_masks[i_local];
        const unsigned long long exclusion_mask = exclusion_masks[i_local];

#pragma unroll
        for (int split_j_lane = 0; split_j_lane < kClusteredSplitJClusterSize;
             split_j_lane += 1)
        {
            const int j_lane =
                split_id * kClusteredSplitJClusterSize + split_j_lane;
            const bool valid_j =
                (valid_mask_j & (1u << static_cast<unsigned int>(j_lane))) !=
                0u;
            const bool local_j =
                (local_mask_j & (1u << static_cast<unsigned int>(j_lane))) !=
                0u;
#pragma unroll
            for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
            {
                const bool local_i =
                    (local_mask_i &
                     (1u << static_cast<unsigned int>(i_lane))) != 0u;
                bool allow_pair = valid_j && local_i;
                if (allow_pair &&
                    !Clustered_Local_I_Owns_Pair(cluster_i, i_lane, cluster_j,
                                                 j_lane, local_j))
                {
                    allow_pair = false;
                }
                if (allow_pair &&
                    (exclusion_mask &
                     (1ull << static_cast<unsigned int>(
                          i_lane * kClusteredClusterSize + j_lane))) != 0ull)
                {
                    allow_pair = false;
                }
                if (allow_pair)
                {
                    source->pair_exclusion_words[split_j_lane *
                                                     kClusteredClusterSize +
                                                 i_lane] |= source_imask_bit;
                }
            }
        }
    }
}

static __device__ __forceinline__ unsigned int
Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift(
    const int split_id, const unsigned int record_imask,
    const unsigned int valid_mask_j, const VECTOR pair_shift,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR* crd,
    const float* shared_i_center_x, const float* shared_i_center_y,
    const float* shared_i_center_z, const VECTOR center_j,
    const int shared_i_atom_ids[kClusteredMaxSuperClusterClusters]
                               [kClusteredClusterSize],
    const int* shared_j_atom_ids, const unsigned int* shared_i_local_masks,
    const float cutoff_sq)
{
    if (record_imask == 0u || crd == NULL)
    {
        return 0u;
    }
    if (cutoff_sq <= 0.0f)
    {
        return record_imask;
    }

    const int j_lane_base = split_id * kClusteredSplitJClusterSize;
    unsigned int pruned_imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int i_local_bit = 1u
                                         << static_cast<unsigned int>(i_local);
        if ((record_imask & i_local_bit) == 0u)
        {
            continue;
        }
        const unsigned int local_mask_i = shared_i_local_masks[i_local];
        if (local_mask_i == 0u)
        {
            continue;
        }

        bool any_in_range = false;
#pragma unroll
        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            if ((local_mask_i & (1u << static_cast<unsigned int>(i_lane))) ==
                0u)
            {
                continue;
            }
            const int atom_i = shared_i_atom_ids[i_local][i_lane];
            if (atom_i < 0)
            {
                continue;
            }
            const VECTOR r_i = crd[atom_i];
            const VECTOR center_i = {shared_i_center_x[i_local],
                                     shared_i_center_y[i_local],
                                     shared_i_center_z[i_local]};
            const VECTOR shifted_i = Shift_Clustered_Atom_Into_Sorted_XQ_Frame(
                r_i, center_i, cell, rcell);
#pragma unroll
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
            {
                const int j_lane = j_lane_base + split_j_lane;
                if ((valid_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) == 0u)
                {
                    continue;
                }
                const int atom_j = shared_j_atom_ids[j_lane];
                if (atom_j < 0)
                {
                    continue;
                }
                const VECTOR shifted_j =
                    Shift_Clustered_Atom_Into_Sorted_XQ_Frame(
                        crd[atom_j], center_j, cell, rcell);
                const float dr_x = shifted_j.x - shifted_i.x - pair_shift.x;
                const float dr_y = shifted_j.y - shifted_i.y - pair_shift.y;
                const float dr_z = shifted_j.z - shifted_i.z - pair_shift.z;
                const float dr2 = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;
                if (dr2 < cutoff_sq && dr2 != 0.0f)
                {
                    any_in_range = true;
                    break;
                }
            }
            if (any_in_range)
            {
                break;
            }
        }
        if (any_in_range)
        {
            pruned_imask |= i_local_bit;
        }
    }
    return pruned_imask;
}

static __device__ __forceinline__ unsigned int
Prune_Gmxpacked_Record_Stream_Source_Imask(
    const int split_id, const unsigned int record_imask,
    const unsigned int valid_mask_j, const int shift_id, const LTMatrix3 cell,
    const LTMatrix3 rcell, const VECTOR* crd, const float* shared_i_center_x,
    const float* shared_i_center_y, const float* shared_i_center_z,
    const VECTOR center_j,
    const int shared_i_atom_ids[kClusteredMaxSuperClusterClusters]
                               [kClusteredClusterSize],
    const int* shared_j_atom_ids, const unsigned int* shared_i_local_masks,
    const float cutoff_sq)
{
    return Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift(
        split_id, record_imask, valid_mask_j,
        Shift_Vector_From_Id(shift_id, cell), cell, rcell, crd,
        shared_i_center_x, shared_i_center_y, shared_i_center_z, center_j,
        shared_i_atom_ids, shared_j_atom_ids, shared_i_local_masks, cutoff_sq);
}

static __host__ __device__ __forceinline__ VECTOR
Wrap_To_Box_Fractional(VECTOR crd, LTMatrix3 rcell)
{
    VECTOR frac = crd * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ uint32_t
Quantize_Unit_Coordinate(float value, int bits)
{
    if (bits <= 0)
    {
        return 0;
    }
    const uint32_t grid = 1u << bits;
    float clamped = fmaxf(0.0f, fminf(0.99999994f, value));
    uint32_t coord = static_cast<uint32_t>(clamped * static_cast<float>(grid));
    if (coord >= grid)
    {
        coord = grid - 1;
    }
    return coord;
}

static __host__ __device__ __forceinline__ VECTOR
Wrap_Unit_Coordinate(VECTOR frac)
{
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ VECTOR
Periodic_Unit_Displacement(VECTOR a, VECTOR b)
{
    VECTOR dr = a - b;
    dr.x -= floorf(dr.x + 0.5f);
    dr.y -= floorf(dr.y + 0.5f);
    dr.z -= floorf(dr.z + 0.5f);
    return dr;
}

static __host__ __device__ __forceinline__ cstone::Vec3<float> To_Cstone_Vec(
    VECTOR v)
{
    return {v.x, v.y, v.z};
}

static __host__ __device__ __forceinline__ VECTOR
Fractional_Cutoff_Pad(float cutoff, LTMatrix3 rcell)
{
    return {cutoff * sqrtf(rcell.a11 * rcell.a11 + rcell.a21 * rcell.a21 +
                           rcell.a31 * rcell.a31),
            cutoff * sqrtf(rcell.a22 * rcell.a22 + rcell.a32 * rcell.a32),
            cutoff * fabsf(rcell.a33)};
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Cornerstone_Node_Overlaps_Shifted_Box(KeyType prefix, VECTOR target_center,
                                      VECTOR target_size, int shift_id)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox = cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    // A half-size of 0.5 represents a periodic reach spanning the complete
    // unit interval on that axis.  It cannot be conservatively represented by
    // one open-boundary image after wrapping.  Keep the tree traversal broad;
    // the fixed-shift cluster AABB screen remains the exact culling stage.
    constexpr float kFullPeriodicReach = 0.5f - 1.0e-6f;
    if (target_size.x >= kFullPeriodicReach ||
        target_size.y >= kFullPeriodicReach ||
        target_size.z >= kFullPeriodicReach)
    {
        return true;
    }
    const VECTOR shifted_center =
        target_center + Shift_Fractional_From_Id(shift_id);
    return cstone::overlap(node_center, node_size,
                           To_Cstone_Vec(shifted_center),
                           To_Cstone_Vec(target_size), unit_box);
}

static __host__ __device__ __forceinline__ bool Exclusion_List_Contains(
    int atom_i, int atom_j, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers)
{
    const int exclude_start = excluded_list_start[atom_i];
    const int exclude_count = excluded_numbers[atom_i];
    for (int k = 0; k < exclude_count; k += 1)
    {
        if (excluded_list[exclude_start + k] == atom_j)
        {
            return true;
        }
    }
    return false;
}

static __device__ __forceinline__ bool Cached_Clusters_May_Share_Molecule(
    const int* molecule_ids_i, const int* molecule_ids_j, uint64_t signature_i,
    uint64_t signature_j, unsigned int local_mask_i, unsigned int valid_mask_j,
    int cluster_size, bool has_molecule_metadata)
{
    if (!has_molecule_metadata)
    {
        return true;
    }
    if ((signature_i & signature_j) == 0ull)
    {
        return false;
    }
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int mol_i = molecule_ids_i[lane_i];
        if (mol_i < 0)
        {
            return true;
        }
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int mol_j = molecule_ids_j[lane_j];
            if (mol_j < 0)
            {
                return true;
            }
            if (mol_i == mol_j)
            {
                return true;
            }
        }
    }
    return false;
}

static __device__ __forceinline__ unsigned long long
Build_Exclusion_Mask_From_Cached_Atoms(
    const int* atom_ids_i, const int* atom_ids_j, const int* molecule_ids_i,
    const int* molecule_ids_j, uint64_t signature_i, uint64_t signature_j,
    bool has_molecule_metadata, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size, int local_atom_numbers,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_numbers)
{
    if (!Cached_Clusters_May_Share_Molecule(
            molecule_ids_i, molecule_ids_j, signature_i, signature_j,
            local_mask_i, valid_mask_j, cluster_size, has_molecule_metadata))
    {
        return 0ull;
    }

    unsigned long long mask = 0ull;
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int atom_i = atom_ids_i[lane_i];
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int atom_j = atom_ids_j[lane_j];
            bool excluded =
                Exclusion_List_Contains(atom_i, atom_j, excluded_list_start,
                                        excluded_list, excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded =
                    Exclusion_List_Contains(atom_j, atom_i, excluded_list_start,
                                            excluded_list, excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}
