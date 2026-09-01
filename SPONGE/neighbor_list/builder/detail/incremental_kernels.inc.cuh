static __global__ void Check_Clustered_Rebuild(
    const int atom_numbers, const VECTOR* crd, const VECTOR* cached_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, int* need_rebuild,
    const float permit_square)
{
    SIMPLE_DEVICE_FOR(tid, atom_numbers)
    {
        const VECTOR dr =
            Get_Periodic_Displacement(crd[tid], cached_crd[tid], cell, rcell);
        if (dr * dr > permit_square)
        {
            need_rebuild[0] = 1;
        }
    }
}

static __global__ void Atomic_Max_Clustered_Anchor_Displacement_Sq(
    const int atom_numbers, const VECTOR* crd, const VECTOR* anchor_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, int* max_displacement_sq_bits)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        const VECTOR current = crd[atom_i];
        const VECTOR anchor = anchor_crd[atom_i];
        if (!isfinite(current.x) || !isfinite(current.y) ||
            !isfinite(current.z) || !isfinite(anchor.x) ||
            !isfinite(anchor.y) || !isfinite(anchor.z))
        {
#ifdef USE_CPU
            continue;
#else
            return;
#endif
        }
        const VECTOR dr =
            Get_Periodic_Displacement(current, anchor, cell, rcell);
        const float displacement_square = dr * dr;
        if (isfinite(displacement_square) && displacement_square >= 0.0f)
        {
            atomicMax(max_displacement_sq_bits,
                      __float_as_int(displacement_square));
        }
    }
}

static __global__ void Mark_Clustered_Incremental_Dirty_Clusters(
    const int atom_numbers, const int cluster_size, const int cluster_numbers,
    const int super_cluster_numbers, const VECTOR* crd,
    const VECTOR* cached_crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int* permutation, const int* cluster_to_supercluster,
    const float permit_square, int* dirty_atoms, int* dirty_clusters,
    int* dirty_superclusters)
{
    SIMPLE_DEVICE_FOR(sorted_idx, atom_numbers)
    {
        const int atom = permutation[sorted_idx];
        if (atom >= 0 && atom < atom_numbers)
        {
            const VECTOR dr = Get_Periodic_Displacement(
                crd[atom], cached_crd[atom], cell, rcell);
            if (dr * dr > permit_square)
            {
                dirty_atoms[atom] = 1;
                const int cluster = sorted_idx / cluster_size;
                if (cluster >= 0 && cluster < cluster_numbers)
                {
                    dirty_clusters[cluster] = 1;
                    const int supercluster = cluster_to_supercluster[cluster];
                    if (supercluster >= 0 &&
                        supercluster < super_cluster_numbers)
                    {
                        dirty_superclusters[supercluster] = 1;
                    }
                }
            }
        }
    }
}

static __global__ void Mark_Gmxpacked_Incremental_Dirty_I_Candidates(
    const int candidate_sci_numbers, const int candidate_super_id_numbers,
    const int super_cluster_numbers,
    const int* candidate_sci_supercluster_ids, const int* dirty_superclusters,
    int* dirty_i_candidate_sci, int* dirty_candidate_sci)
{
    SIMPLE_DEVICE_FOR(candidate_sci, candidate_sci_numbers)
    {
        const int sci_base = candidate_sci / kClusteredShiftCount;
        int dirty = 0;
        if (sci_base >= 0 && sci_base < candidate_super_id_numbers)
        {
            const int super_i = candidate_sci_supercluster_ids[sci_base];
            dirty = (super_i >= 0 && super_i < super_cluster_numbers &&
                     dirty_superclusters[super_i] != 0)
                        ? 1
                        : 0;
        }
        dirty_i_candidate_sci[candidate_sci] = dirty;
        dirty_candidate_sci[candidate_sci] = dirty;
    }
}

static __global__ void Mark_Gmxpacked_Incremental_Dirty_J_Candidates_Parallel(
    const int candidate_sci_numbers, const int candidate_leaf_numbers,
    const int cluster_numbers, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const int* dirty_clusters,
    int* dirty_candidate_sci)
{
    const int candidate_sci = blockIdx.x;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }
    __shared__ int shared_dirty;
    if (threadIdx.x == 0)
    {
        shared_dirty = dirty_candidate_sci[candidate_sci] != 0 ? 1 : 0;
    }
    __syncthreads();
    if (shared_dirty == 0)
    {
        const int candidate_begin = candidate_leaf_offsets[candidate_sci];
        const int candidate_end = candidate_leaf_offsets[candidate_sci + 1];
        bool local_dirty = false;
        for (int candidate_idx = candidate_begin + threadIdx.x;
             candidate_idx < candidate_end && !local_dirty;
             candidate_idx += blockDim.x)
        {
            if (candidate_idx < 0 || candidate_idx >= candidate_leaf_numbers)
            {
                continue;
            }
            const int leaf = candidate_leaf_ids[candidate_idx];
            if (leaf < 0)
            {
                continue;
            }
            const int cluster_begin = leaf_cluster_starts[leaf];
            const int cluster_end = leaf_cluster_ends[leaf];
            for (int cluster = cluster_begin; cluster < cluster_end;
                 cluster += 1)
            {
                if (cluster >= 0 && cluster < cluster_numbers &&
                    dirty_clusters[cluster] != 0)
                {
                    local_dirty = true;
                    atomicExch(&shared_dirty, 1);
                    break;
                }
            }
            if (shared_dirty != 0)
            {
                break;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && shared_dirty != 0)
    {
        dirty_candidate_sci[candidate_sci] = 1;
    }
}
