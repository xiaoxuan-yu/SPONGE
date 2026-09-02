#include "workspace.h"

void LJClusteredWorkspace::Initialize(
    ClusteredNeighborProvider* provider)
{
    provider_ = provider;
    coordinate_gather_step = -1;
    coordinate_gather_count_this_step = 0;
    coordinate_gather_count_total = 0;
    gathered_flavor_ = GatherFlavor::NONE;
    gathered_provider_incarnation_ = 0;
    gathered_geometry_generation_ = 0;
}

bool LJClusteredWorkspace::IsBoundTo(
    const ClusteredNeighborProvider* provider) const
{
    return provider_ == provider;
}

void LJClusteredWorkspace::Clear()
{
#ifndef USE_CPU
    if (provider_ != NULL && provider_->IsInitialized())
    {
        provider_->BindWorkingDevice();
    }
#endif
    Free_Single_Device_Pointer((void**)&d_sorted_atom_ids);
    Free_Single_Device_Pointer((void**)&d_sorted_xq);
    Free_Single_Device_Pointer((void**)&d_sorted_lj_type);
    Free_Single_Device_Pointer((void**)&d_sorted_lj_comb);
    Free_Single_Device_Pointer((void**)&d_sorted_frc);
    Free_Single_Device_Pointer((void**)&d_sorted_soft_crd);
    sorted_atom_ids_capacity = 0;
    sorted_xq_capacity = 0;
    sorted_lj_type_capacity = 0;
    sorted_lj_comb_capacity = 0;
    sorted_frc_capacity = 0;
    sorted_soft_crd_capacity = 0;
    payload_gather_time_recorder = NULL;
    direct_kernel_time_recorder = NULL;
    gmxpacked_force_scratch_memset_time_recorder = NULL;
    gmxpacked_kernel_launch_time_recorder = NULL;
    gmxpacked_sorted_force_scatter_time_recorder = NULL;
    coordinate_gather_step = -1;
    coordinate_gather_count_this_step = 0;
    coordinate_gather_count_total = 0;
    gathered_flavor_ = GatherFlavor::NONE;
    gathered_provider_incarnation_ = 0;
    gathered_geometry_generation_ = 0;
    provider_ = NULL;
}
