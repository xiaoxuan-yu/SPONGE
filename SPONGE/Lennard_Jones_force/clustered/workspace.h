#pragma once

#include "../../neighbor_list/provider/provider.h"

struct VECTOR_LJ;
struct VECTOR_LJ_SOFT_TYPE;

class LJClusteredWorkspace
{
public:
    TIME_RECORDER* payload_gather_time_recorder = NULL;
    TIME_RECORDER* direct_kernel_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_force_scratch_memset_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_kernel_launch_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_sorted_force_scatter_time_recorder = NULL;
    int coordinate_gather_step = -1;
    int coordinate_gather_count_this_step = 0;
    long long coordinate_gather_count_total = 0;

    int sorted_atom_ids_capacity = 0;
    int* d_sorted_atom_ids = NULL;
    int sorted_xq_capacity = 0;
    float4* d_sorted_xq = NULL;
    int sorted_lj_type_capacity = 0;
    int* d_sorted_lj_type = NULL;
    int sorted_lj_comb_capacity = 0;
    float2* d_sorted_lj_comb = NULL;
    int sorted_frc_capacity = 0;
    VECTOR* d_sorted_frc = NULL;
    int sorted_soft_crd_capacity = 0;
    VECTOR_LJ_SOFT_TYPE* d_sorted_soft_crd = NULL;

    LJClusteredWorkspace() = default;
    LJClusteredWorkspace(const LJClusteredWorkspace&) = delete;
    LJClusteredWorkspace& operator=(const LJClusteredWorkspace&) = delete;

    void Initialize(ClusteredNeighborProvider* provider);
    bool IsBoundTo(const ClusteredNeighborProvider* provider) const;
    void Gather_Plain(const VECTOR* crd, const float* charge,
                      const VECTOR_LJ* lj_type_src, LTMatrix3 cell,
                      LTMatrix3 rcell, const float2* lj_ab_packed = NULL);
    void Gather_Plain(const VECTOR_LJ* src, LTMatrix3 cell, LTMatrix3 rcell,
                      const float2* lj_ab_packed = NULL);
    void Gather_Soft_Core(const VECTOR_LJ_SOFT_TYPE* src, LTMatrix3 cell,
                          LTMatrix3 rcell);
    bool Plain_Gather_Ready_For_Current_Step() const;
    bool Soft_Core_Gather_Ready_For_Current_Step() const;
    void Clear();

private:
    enum class GatherFlavor
    {
        NONE,
        PLAIN,
        SOFT_CORE
    };

    bool Gather_Ready_For_Current_Step(GatherFlavor flavor) const;

    ClusteredNeighborProvider* provider_ = NULL;
    GatherFlavor gathered_flavor_ = GatherFlavor::NONE;
    uint64_t gathered_provider_incarnation_ = 0;
    uint64_t gathered_geometry_generation_ = 0;
};
