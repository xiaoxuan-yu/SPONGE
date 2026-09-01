#pragma once

// Internal clustered-builder geometry primitives. Include only after the
// backend mask primitives required by the subgroup helpers are visible.

static __host__ __device__ __forceinline__ VECTOR
Wrap_Clustered_Center_Fractional(const VECTOR center, const LTMatrix3 rcell)
{
    VECTOR frac = center * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ VECTOR
Clustered_Fractional_Extent(const VECTOR extent, const LTMatrix3 rcell)
{
    return {
        fabsf(extent.x * rcell.a11) + fabsf(extent.y * rcell.a21) +
            fabsf(extent.z * rcell.a31),
        fabsf(extent.y * rcell.a22) + fabsf(extent.z * rcell.a32),
        fabsf(extent.z * rcell.a33)};
}

static __host__ __device__ __forceinline__ void
Store_Current_Cluster_Fractional_Geometry(
    const int cluster_i, const VECTOR center, const VECTOR extent,
    const LTMatrix3 rcell, VECTOR* cluster_fractional_centers,
    VECTOR* cluster_fractional_extents)
{
    cluster_fractional_centers[cluster_i] =
        Wrap_Clustered_Center_Fractional(center, rcell);
    cluster_fractional_extents[cluster_i] =
        Clustered_Fractional_Extent(extent, rcell);
}

static __host__ __device__ __forceinline__ int Clamp_Shift_Component(int shift)
{
    return shift < -1 ? -1 : (shift > 1 ? 1 : shift);
}

static __host__ __device__ __forceinline__ int Encode_Shift_Id(int sx, int sy,
                                                               int sz)
{
    return (Clamp_Shift_Component(sx) + 1) * 9 +
           (Clamp_Shift_Component(sy) + 1) * 3 +
           (Clamp_Shift_Component(sz) + 1);
}

static __host__ __device__ __forceinline__ VECTOR Shift_Fractional_From_Id(
    int shift_id)
{
    return {
        static_cast<float>(shift_id / 9 - 1),
        static_cast<float>((shift_id % 9) / 3 - 1),
        static_cast<float>(shift_id % 3 - 1)};
}

static __host__ __device__ __forceinline__ VECTOR Shift_Vector_From_Id(
    int shift_id, LTMatrix3 cell)
{
    const VECTOR frac_shift = Shift_Fractional_From_Id(shift_id);
    return frac_shift * cell;
}

static __host__ __device__ __forceinline__ int Determine_Shift_Id(
    VECTOR frac_i, VECTOR frac_j)
{
    const VECTOR dfrac = frac_j - frac_i;
    const int sx = static_cast<int>(floorf(dfrac.x + 0.5f));
    const int sy = static_cast<int>(floorf(dfrac.y + 0.5f));
    const int sz = static_cast<int>(floorf(dfrac.z + 0.5f));
    return Encode_Shift_Id(sx, sy, sz);
}

static __host__ __device__ __forceinline__ int Determine_Cluster_Pair_Shift_Id(
    VECTOR center_i, VECTOR center_j, LTMatrix3 rcell)
{
    VECTOR frac_i = center_i * rcell;
    frac_i.x -= floorf(frac_i.x);
    frac_i.y -= floorf(frac_i.y);
    frac_i.z -= floorf(frac_i.z);
    VECTOR frac_j = center_j * rcell;
    frac_j.x -= floorf(frac_j.x);
    frac_j.y -= floorf(frac_j.y);
    frac_j.z -= floorf(frac_j.z);
    return Determine_Shift_Id(frac_i, frac_j);
}

static __host__ __device__ __forceinline__ VECTOR
Shift_Clustered_Atom_Into_Sorted_XQ_Frame(const VECTOR atom_crd,
                                          const VECTOR cluster_center,
                                          const LTMatrix3 cell,
                                          const LTMatrix3 rcell)
{
    return cluster_center +
           Get_Periodic_Displacement(atom_crd, cluster_center, cell, rcell);
}

static __host__ __device__ __forceinline__ VECTOR Fractional_Extent_Pad(
    VECTOR extent, LTMatrix3 rcell)
{
    return {
        fabsf(extent.x * rcell.a11) + fabsf(extent.y * rcell.a21) +
            fabsf(extent.z * rcell.a31),
        fabsf(extent.y * rcell.a22) + fabsf(extent.z * rcell.a32),
        fabsf(extent.z * rcell.a33)};
}

static __host__ __device__ __forceinline__ bool
Cluster_Aabb_Overlaps_Shifted_CutoffSq(VECTOR center_i, VECTOR extent_i,
                                       VECTOR center_j, VECTOR extent_j,
                                       float cutoff_sq, VECTOR shift_vec);

static __host__ __device__ __forceinline__ bool Cluster_Aabb_Overlaps_Shifted(
    VECTOR center_i, VECTOR extent_i, VECTOR center_j, VECTOR extent_j,
    float cutoff, VECTOR shift_vec)
{
    return Cluster_Aabb_Overlaps_Shifted_CutoffSq(
        center_i, extent_i, center_j, extent_j, cutoff * cutoff, shift_vec);
}

static __host__ __device__ __forceinline__ bool
Cluster_Aabb_Overlaps_Shifted_CutoffSq(VECTOR center_i, VECTOR extent_i,
                                       VECTOR center_j, VECTOR extent_j,
                                       float cutoff_sq, VECTOR shift_vec)
{
    const VECTOR dr = center_j - (center_i + shift_vec);
    const float gap_x =
        fmaxf(fabsf(dr.x) - (extent_i.x + extent_j.x), 0.0f);
    const float gap_y =
        fmaxf(fabsf(dr.y) - (extent_i.y + extent_j.y), 0.0f);
    const float gap_z =
        fmaxf(fabsf(dr.z) - (extent_i.z + extent_j.z), 0.0f);
    return gap_x * gap_x + gap_y * gap_y + gap_z * gap_z <= cutoff_sq;
}

static __host__ __device__ __forceinline__ bool
Clustered_Valid_Lanes_Are_All_Local(unsigned int valid_mask,
                                    unsigned int local_mask)
{
    return valid_mask != 0u && (valid_mask & ~local_mask) == 0u;
}

static __host__ __device__ __forceinline__ unsigned int
Build_Fixed_Shift_Cluster_I_Mask(
    int cluster_i_start, int cluster_i_end, int cluster_j, int fixed_shift_id,
    float cutoff, VECTOR shift_vec, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents)
{
    const VECTOR center_j = cluster_centers[cluster_j];
    const VECTOR extent_j = cluster_extents[cluster_j];
    unsigned int i_mask = 0u;
    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const int i_local = cluster_i - cluster_i_start;
        if (cluster_local_masks[cluster_i] == 0u)
        {
            continue;
        }
        if (cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j &&
            Clustered_Valid_Lanes_Are_All_Local(
                cluster_valid_masks[cluster_j],
                cluster_local_masks[cluster_j]))
        {
            continue;
        }
        if (Cluster_Aabb_Overlaps_Shifted(
                cluster_centers[cluster_i], cluster_extents[cluster_i],
                center_j, extent_j, cutoff, shift_vec))
        {
            i_mask |= (1u << static_cast<unsigned int>(i_local));
        }
    }
    return i_mask;
}

static __host__ __device__ __forceinline__ bool
Leaf_Has_Fixed_Shift_Candidate_Overlap(
    int cluster_i_start, int cluster_i_end, int leaf_cluster_start,
    int leaf_cluster_end, int fixed_shift_id, float cutoff, VECTOR shift_vec,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents)
{
    for (int cluster_j = leaf_cluster_start; cluster_j < leaf_cluster_end;
         cluster_j += 1)
    {
        if (cluster_valid_masks[cluster_j] == 0u)
        {
            continue;
        }
        if (Build_Fixed_Shift_Cluster_I_Mask(
                cluster_i_start, cluster_i_end, cluster_j, fixed_shift_id,
                cutoff, shift_vec, cluster_valid_masks, cluster_local_masks,
                cluster_centers, cluster_extents) != 0u)
        {
            return true;
        }
    }
    return false;
}

static __device__ __forceinline__ bool
Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup(
    int cluster_i_start, int cluster_i_end, int leaf_cluster_start,
    int leaf_cluster_end, int fixed_shift_id, float cutoff, VECTOR shift_vec,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, int sublane, device_mask_t subgroup_mask)
{
    bool leaf_overlap = false;
    for (int cluster_j = leaf_cluster_start; cluster_j < leaf_cluster_end;
         cluster_j += 1)
    {
        bool lane_overlap = false;
        if (cluster_valid_masks[cluster_j] != 0u)
        {
            const int cluster_i = cluster_i_start + sublane;
            if (cluster_i < cluster_i_end &&
                cluster_local_masks[cluster_i] != 0u)
            {
                const bool reverse_local_cluster_pair =
                    cluster_j >= cluster_i_start &&
                    cluster_j < cluster_i_end && cluster_i > cluster_j &&
                    Clustered_Valid_Lanes_Are_All_Local(
                        cluster_valid_masks[cluster_j],
                        cluster_local_masks[cluster_j]);
                if (!reverse_local_cluster_pair)
                {
                    lane_overlap = Cluster_Aabb_Overlaps_Shifted(
                        cluster_centers[cluster_i],
                        cluster_extents[cluster_i],
                        cluster_centers[cluster_j],
                        cluster_extents[cluster_j], cutoff, shift_vec);
                }
            }
        }
        if (deviceBallot(subgroup_mask, lane_overlap) != 0u)
        {
            leaf_overlap = true;
            break;
        }
    }
    return leaf_overlap;
}
