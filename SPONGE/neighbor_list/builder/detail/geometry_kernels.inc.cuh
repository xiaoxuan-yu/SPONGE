static __global__ void Build_Cornerstone_Sort_Keys(const int atom_numbers,
                                                   const VECTOR* crd,
                                                   const LTMatrix3 rcell,
                                                   uint64_t* keys,
                                                   int* permutation)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        const VECTOR frac = Wrap_To_Box_Fractional(crd[i], rcell);
        constexpr int bits = cstone::maxTreeLevel<CornerstoneKey>{};
        const uint32_t x = Quantize_Unit_Coordinate(frac.x, bits);
        const uint32_t y = Quantize_Unit_Coordinate(frac.y, bits);
        const uint32_t z = Quantize_Unit_Coordinate(frac.z, bits);
        keys[i] = cstone::iHilbert<CornerstoneKey>(x, y, z);
        permutation[i] = i;
    }
}

static __global__ void Copy_UInt_To_Int(const int count, const unsigned* src,
                                        int* dest)
{
    SIMPLE_DEVICE_FOR(i, count) { dest[i] = static_cast<int>(src[i]); }
}

static __global__ void Build_Local_Atom_To_Molecule_Map(
    const int atom_numbers, const int global_atom_numbers,
    const int* atom_local, const int* global_atom_to_molecule,
    int* local_atom_to_molecule)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        const int global_atom = atom_local != NULL ? atom_local[i] : i;
        int molecule_id = -1;
        if (global_atom >= 0 && global_atom < global_atom_numbers &&
            global_atom_to_molecule != NULL)
        {
            molecule_id = global_atom_to_molecule[global_atom];
        }
        local_atom_to_molecule[i] = molecule_id;
    }
}

static __global__ void Build_Cluster_Molecule_Metadata(
    const int cluster_numbers, const int cluster_size, const int* permutation,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const int* atom_to_molecule, uint64_t* cluster_molecule_signatures,
    int* cluster_molecule_ids)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int atom_start = cluster_offsets[cluster_i];
        const unsigned int valid_mask = cluster_valid_masks[cluster_i];
        uint64_t signature = 0ull;
        for (int lane = 0; lane < cluster_size; lane += 1)
        {
            int molecule_id = -1;
            if ((valid_mask & (1u << lane)) != 0u && atom_to_molecule != NULL)
            {
                const int atom = permutation[atom_start + lane];
                molecule_id = atom_to_molecule[atom];
                if (molecule_id >= 0)
                {
                    signature |=
                        1ull << (static_cast<unsigned int>(molecule_id) & 63u);
                }
            }
            cluster_molecule_ids[cluster_i * cluster_size + lane] = molecule_id;
        }
        cluster_molecule_signatures[cluster_i] = signature;
    }
}

static __global__ void Build_Global_Cluster_Metadata(
    const int cluster_numbers, const int total_atom_numbers,
    const int direct_local_atom_numbers, const int cluster_size,
    const int* permutation, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, int* cluster_offsets,
    unsigned int* cluster_valid_masks, unsigned int* cluster_local_masks,
    VECTOR* cluster_centers, VECTOR* cluster_extents,
    VECTOR* cluster_fractional_centers, VECTOR* cluster_fractional_extents,
    float* cluster_radii)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int raw_start = cluster_i * cluster_size;
        const int start = IntMin(total_atom_numbers, raw_start);
        const int end = IntMin(total_atom_numbers, raw_start + cluster_size);
        const int count = IntMax(0, end - start);
        unsigned int valid_mask = 0u;
        unsigned int local_mask = 0u;
        VECTOR center = {0.0f, 0.0f, 0.0f};
        VECTOR extent = {0.0f, 0.0f, 0.0f};
        float radius = 0.0f;

        cluster_offsets[cluster_i] = start;
        if (count > 0)
        {
            const VECTOR anchor = crd[permutation[start]];
            for (int lane = 0; lane < count; lane += 1)
            {
                valid_mask |= (1u << lane);
                const int atom_index = permutation[start + lane];
                if (atom_index < direct_local_atom_numbers)
                {
                    local_mask |= (1u << lane);
                }
                const VECTOR pos = crd[atom_index];
                center = center + (anchor + Get_Periodic_Displacement(
                                                pos, anchor, cell, rcell));
            }
            center = (1.0f / static_cast<float>(count)) * center;
            center = Get_Periodic_Coordinate(center, cell, rcell);
            for (int lane = 0; lane < count; lane += 1)
            {
                const int atom_index = permutation[start + lane];
                const VECTOR pos = crd[atom_index];
                const VECTOR dr =
                    Get_Periodic_Displacement(pos, center, cell, rcell);
                extent.x = fmaxf(extent.x, fabsf(dr.x));
                extent.y = fmaxf(extent.y, fabsf(dr.y));
                extent.z = fmaxf(extent.z, fabsf(dr.z));
                radius = fmaxf(radius, norm3df(dr.x, dr.y, dr.z));
            }
        }

        cluster_valid_masks[cluster_i] = valid_mask;
        cluster_local_masks[cluster_i] = local_mask;
        cluster_centers[cluster_i] = center;
        cluster_extents[cluster_i] = extent;
        Store_Current_Cluster_Fractional_Geometry(
            cluster_i, center, extent, rcell, cluster_fractional_centers,
            cluster_fractional_extents);
        cluster_radii[cluster_i] = radius;
        if (cluster_i == cluster_numbers - 1)
        {
            cluster_offsets[cluster_numbers] = end;
        }
    }
}

static __global__ void Build_Leaf_Cluster_Ranges(const int leaf_numbers,
                                                 const int cluster_size,
                                                 const int* leaf_atom_offsets,
                                                 int* leaf_cluster_starts,
                                                 int* leaf_cluster_ends)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const int cluster_start = atom_start / cluster_size;
        const int cluster_end =
            atom_end > atom_start ? (atom_end + cluster_size - 1) / cluster_size
                                  : cluster_start;
        leaf_cluster_starts[leaf_i] = cluster_start;
        leaf_cluster_ends[leaf_i] = cluster_end;
    }
}

// Compute two rebuild-time bounds in one pass: the exact maximum per-leaf
// cluster span used by the subgroup dedup scan, and the maximum fractional
// extent of any cluster from its center. A cluster has at most eight atoms in
// one octree leaf, so its center-to-atom extent is at most 7/8 of the leaf
// width along each fractional axis.
static __global__ void Reduce_Max_Leaf_Cluster_Span(
    const int leaf_numbers, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const uint64_t* leaves, int* d_max_span,
    int* d_max_extent_bits)
{
    int local_max = 0;
    int local_max_extent_bits = 0;
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int span =
            leaf_cluster_ends[leaf_i] - leaf_cluster_starts[leaf_i];
        if (span > local_max)
        {
            local_max = span;
        }
        const unsigned int level =
            cstone::treeLevel(leaves[leaf_i + 1] - leaves[leaf_i]);
        const float fractional_extent_bound =
            0.875f / static_cast<float>(1u << level);
        const int fractional_extent_bits =
            __float_as_int(fractional_extent_bound);
        if (fractional_extent_bits > local_max_extent_bits)
        {
            local_max_extent_bits = fractional_extent_bits;
        }
    }
    atomicMax(d_max_span, local_max);
    atomicMax(d_max_extent_bits, local_max_extent_bits);
}

static __global__ void Build_Fixed_Group_Offsets(const int offset_numbers,
                                                 const int group_size,
                                                 const int total_numbers,
                                                 int* offsets)
{
    SIMPLE_DEVICE_FOR(group_i, offset_numbers)
    {
        offsets[group_i] = IntMin(group_i * group_size, total_numbers);
    }
}

static __global__ void Build_Cluster_To_Supercluster(
    const int super_cluster_numbers, const int* super_cluster_offsets,
    int* cluster_to_supercluster)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        const int cluster_start = super_cluster_offsets[super_i];
        const int cluster_end = super_cluster_offsets[super_i + 1];
        for (int cluster_i = cluster_start; cluster_i < cluster_end;
             cluster_i += 1)
        {
            cluster_to_supercluster[cluster_i] = super_i;
        }
    }
}

static __global__ void Build_Supercluster_Metadata(
    const int super_cluster_numbers, const int super_cluster_clusters,
    const float cutoff, const LTMatrix3 rcell, const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, int* super_cluster_has_local,
    VECTOR* super_cluster_centers, VECTOR* super_cluster_sizes)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        const int cluster_start = super_cluster_offsets[super_i];
        const int cluster_end = super_cluster_offsets[super_i + 1];
        int cluster_count = 0;
        int has_local = 0;
        VECTOR anchor_frac = {0.0f, 0.0f, 0.0f};
        VECTOR frac_sum = {0.0f, 0.0f, 0.0f};
        VECTOR frac_center = {0.0f, 0.0f, 0.0f};
        VECTOR frac_size = {0.0f, 0.0f, 0.0f};

        for (int cluster_i = cluster_start; cluster_i < cluster_end;
             cluster_i += 1)
        {
            if (cluster_valid_masks[cluster_i] == 0u)
            {
                continue;
            }
            const VECTOR frac =
                Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
            if (cluster_count == 0)
            {
                anchor_frac = frac;
            }
            frac_sum = frac_sum + (anchor_frac + Periodic_Unit_Displacement(
                                                     frac, anchor_frac));
            has_local |= cluster_local_masks[cluster_i] != 0u;
            cluster_count += 1;
        }

        if (cluster_count > 0)
        {
            frac_center = (1.0f / static_cast<float>(cluster_count)) * frac_sum;
            frac_center = Wrap_Unit_Coordinate(frac_center);
            const VECTOR cutoff_pad = Fractional_Cutoff_Pad(cutoff, rcell);
            for (int cluster_i = cluster_start; cluster_i < cluster_end;
                 cluster_i += 1)
            {
                if (cluster_valid_masks[cluster_i] == 0u)
                {
                    continue;
                }
                const VECTOR frac =
                    Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
                const VECTOR dfrac =
                    Periodic_Unit_Displacement(frac, frac_center);
                const VECTOR extent_pad =
                    Fractional_Extent_Pad(cluster_extents[cluster_i], rcell);
                frac_size.x = fmaxf(frac_size.x, fabsf(dfrac.x) + extent_pad.x);
                frac_size.y = fmaxf(frac_size.y, fabsf(dfrac.y) + extent_pad.y);
                frac_size.z = fmaxf(frac_size.z, fabsf(dfrac.z) + extent_pad.z);
            }
            frac_size.x = fminf(0.5f, frac_size.x + cutoff_pad.x);
            frac_size.y = fminf(0.5f, frac_size.y + cutoff_pad.y);
            frac_size.z = fminf(0.5f, frac_size.z + cutoff_pad.z);
        }

        (void)super_cluster_clusters;
        super_cluster_has_local[super_i] = has_local;
        super_cluster_centers[super_i] = frac_center;
        super_cluster_sizes[super_i] = frac_size;
    }
}

static __global__ void Build_Local_Supercluster_Flags(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    int* sci_flags)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        sci_flags[super_i] = super_cluster_has_local[super_i] != 0 ? 1 : 0;
    }
}
