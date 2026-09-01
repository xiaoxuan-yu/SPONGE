#include "candidate_builder.h"
#include "internal.h"

#include "../provider/internal.h"
#include "../provider/runtime.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../../MD_core/MD_core.h"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/csarray.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"

#ifndef USE_CPU
#include "../../third_party/cornerstone_octree/include/cstone/cuda/device_vector.h"
#endif

namespace
{

using CornerstoneKey = uint64_t;
using CornerstoneNodeIndex = cstone::TreeNodeIndex;
using clustered_neighbor_runtime::Reserve_Device_Buffer;
using cstone::rawPtr;

constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
constexpr int kFixedShiftCandidateLeafSubgroupSize =
    kClusteredMaxSuperClusterClusters;
constexpr int kCandidateLeafOnepassInitialLeavesPerCandidateSci = 128;
constexpr long long kCandidateLeafOnepassScratchByteLimit =
    256ll * 1024ll * 1024ll;
constexpr int kCandidateLeafOnepassSlackNumerator = 5;
constexpr int kCandidateLeafOnepassSlackDenominator = 4;
constexpr int kFixedShiftCandidateLeafQueue2DeviceBlocks = 256;
constexpr int kFixedShiftCandidateLeafQueue2TaskSplitDepth = 2;

#include "primitives/geometry.cuh"
#include "detail/pair_mask_primitives.inc.cuh"
#include "detail/candidate_kernels.inc.cuh"

static int Candidate_Leaf_Onepass_Scratch_Int_Streams(bool count_metadata)
{
    return count_metadata ? 4 : 3;
}

static int Candidate_Leaf_Onepass_Max_Record_Capacity(bool count_metadata)
{
    const long long max_by_bytes =
        kCandidateLeafOnepassScratchByteLimit /
        (static_cast<long long>(
             Candidate_Leaf_Onepass_Scratch_Int_Streams(count_metadata)) *
         sizeof(int));
    const long long max_by_int = std::numeric_limits<int>::max();
    return static_cast<int>(max_by_bytes < max_by_int ? max_by_bytes
                                                      : max_by_int);
}

static int Candidate_Leaf_Onepass_Clamp_Record_Capacity(long long capacity,
                                                        bool count_metadata)
{
    if (capacity <= 0)
    {
        return 0;
    }
    const int max_capacity =
        Candidate_Leaf_Onepass_Max_Record_Capacity(count_metadata);
    if (capacity > static_cast<long long>(max_capacity))
    {
        return max_capacity;
    }
    return static_cast<int>(capacity);
}

static long long Candidate_Leaf_Onepass_Ceil_Ratio(int value, int numerator,
                                                   int denominator)
{
    if (value <= 0)
    {
        return 0;
    }
    return (static_cast<long long>(value) * numerator + denominator - 1) /
           denominator;
}

static int Candidate_Leaf_Onepass_Target_Capacity(
    int candidate_sci_numbers, int candidate_leaf_capacity,
    int candidate_leaf_onepass_high_water,
    int candidate_leaf_onepass_record_capacity, bool count_metadata)
{
    long long target = static_cast<long long>(candidate_sci_numbers) *
                       kCandidateLeafOnepassInitialLeavesPerCandidateSci;
    const int observed_capacity =
        IntMax(candidate_leaf_capacity, candidate_leaf_onepass_high_water);
    if (observed_capacity > 0)
    {
        target = std::max(
            target, Candidate_Leaf_Onepass_Ceil_Ratio(
                        observed_capacity, kCandidateLeafOnepassSlackNumerator,
                        kCandidateLeafOnepassSlackDenominator));
    }
    target = std::max(
        target, static_cast<long long>(candidate_leaf_onepass_record_capacity));
    return Candidate_Leaf_Onepass_Clamp_Record_Capacity(target, count_metadata);
}

static void Reserve_Candidate_Leaf_Onepass_Scratch(
    ClusteredNeighborProvider* layout, int capacity, bool count_metadata)
{
    capacity =
        Candidate_Leaf_Onepass_Clamp_Record_Capacity(capacity, count_metadata);
    if (layout == NULL || capacity <= 0)
    {
        return;
    }
    Reserve_Device_Buffer(
        capacity,
        &ClusteredNeighborProviderInternal::Workspace(layout)
             .candidate_leaf_onepass_sci_ids);
    Reserve_Device_Buffer(
        capacity,
        &ClusteredNeighborProviderInternal::Workspace(layout)
             .candidate_leaf_onepass_ranks);
    Reserve_Device_Buffer(
        capacity,
        &ClusteredNeighborProviderInternal::Workspace(layout)
             .candidate_leaf_onepass_leaf_ids);
    if (count_metadata)
    {
        Reserve_Device_Buffer(
            capacity,
            &ClusteredNeighborProviderInternal::Workspace(layout)
                 .candidate_leaf_onepass_prev_running_max_ends);
    }
    ClusteredNeighborProviderInternal::Workspace(layout)
        .candidate_leaf_onepass_record_capacity = capacity;
}

}  // namespace

namespace clustered_neighbor_builder_internal
{

CandidateBuildResult BuildCandidateStage(
    ClusteredNeighborProvider* layout, const VECTOR* commit_cache_crd, LTMatrix3 cell,
    float cutoff, float build_cutoff)
{
    const int super_sci_numbers = ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_numbers;
    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_cluster_stride =
        (ClusteredNeighborProviderInternal::Config(layout).cornerstone_leaf_size +
         2 * ClusteredNeighborProviderInternal::Config(layout).cluster_size - 2) /
        ClusteredNeighborProviderInternal::Config(layout).cluster_size;
    const int candidate_sci_numbers = super_sci_numbers * kClusteredShiftCount;
    const int* candidate_sci_supercluster_ids = NULL;
    const int* candidate_shift_ids = NULL;

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_supercluster_ids);
    Launch_Device_Kernel(Fill_Local_Supercluster_Ids,
                         (ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.has_local.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_supercluster_ids.data);
    candidate_sci_supercluster_ids =
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_supercluster_ids.data;

    Reserve_Device_Buffer(candidate_sci_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts);
    Reserve_Device_Buffer(candidate_sci_numbers + 1,
                          &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets);
    bool candidate_leaf_queue2_ids_materialized = false;
    bool candidate_leaf_queue2_counts_ready = false;
#ifndef USE_CPU
    if (!candidate_leaf_queue2_ids_materialized)
    {
        const int task_split_depth =
            kFixedShiftCandidateLeafQueue2TaskSplitDepth;
        const int max_tasks_per_candidate_sci = task_split_depth <= 1 ? 8 : 64;
        const long long task_capacity_request_ll =
            static_cast<long long>(candidate_sci_numbers) *
            static_cast<long long>(max_tasks_per_candidate_sci);
        const int task_capacity_request =
            task_capacity_request_ll >
                    static_cast<long long>(std::numeric_limits<int>::max())
                ? std::numeric_limits<int>::max()
                : static_cast<int>(task_capacity_request_ll);
        if (task_capacity_request > 0)
        {
            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity =
                task_capacity_request;
            Reserve_Device_Buffer(
                1, &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter);
            Reserve_Device_Buffer(
                1, &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_overflow);
            Reserve_Device_Buffer(
                1, &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_work_cursor);
            Reserve_Device_Buffer(
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity,
                &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids);
            Reserve_Device_Buffer(
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity,
                &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes);
            deviceMemset(
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter.data, 0,
                sizeof(int));
            deviceMemset(
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_overflow.data, 0,
                sizeof(int));
            deviceMemset(ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data, 0,
                         sizeof(int) * candidate_sci_numbers);
            {
                const int task_build_block_size = 128;
                const int task_build_items = candidate_sci_numbers * 8;
                const int task_build_blocks =
                    (task_build_items + task_build_block_size - 1) /
                    task_build_block_size;
                Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
                    task_build_blocks, task_build_block_size,
                    candidate_sci_numbers, candidate_sci_supercluster_ids,
                    ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
                    cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                       ->octree.prefixes),
                    cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                       ->octree.childOffsets),
                    candidate_shift_ids, false,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter.data,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_overflow.data,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids.data,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes.data,
                    task_split_depth);
            }
            int h_queue2_tasks = 0;
            int h_queue2_task_overflow = 0;
            deviceMemcpy(
                &h_queue2_tasks,
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter.data,
                sizeof(int), deviceMemcpyDeviceToHost);
            deviceMemcpy(
                &h_queue2_task_overflow,
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_overflow.data,
                sizeof(int), deviceMemcpyDeviceToHost);
            if (h_queue2_tasks >
                ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity)
            {
                h_queue2_tasks =
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity;
            }
            if (h_queue2_tasks > 1)
            {
                Reserve_Device_Buffer(
                    h_queue2_tasks,
                    &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sort_keys);
                Reserve_Device_Buffer(
                    h_queue2_tasks,
                    &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_pairs);
                {
                    Launch_Device_Kernel(
                        Build_Candidate_Leaf_Queue2_Task_Sort_Keys,
                        (h_queue2_tasks + 255) / 256, 256, 0, NULL,
                        h_queue2_tasks,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sort_keys
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_pairs
                            .data);
                    StableSortU64Pair(
                        layout, h_queue2_tasks,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sort_keys
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_pairs
                            .data);
                    Launch_Device_Kernel(
                        Unpack_Candidate_Leaf_Queue2_Task_Pairs,
                        (h_queue2_tasks + 255) / 256, 256, 0, NULL,
                        h_queue2_tasks,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_pairs.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes
                            .data);
                }
            }
            Reserve_Device_Buffer(
                IntMax(h_queue2_tasks, 1),
                &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_leaf_counts);
            Reserve_Device_Buffer(
                IntMax(h_queue2_tasks + 1, 1),
                &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_leaf_offsets);
            if (h_queue2_tasks > 0)
            {
                deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout)
                                 .candidate_leaf_queue2_task_leaf_counts.data,
                             0, sizeof(int) * h_queue2_tasks);
            }
            int candidate_leaf_numbers_by_task = 0;
            bool candidate_leaf_queue2_fused_used = false;
            if (h_queue2_tasks > 0)
            {
                const int fused_record_capacity =
                    Candidate_Leaf_Onepass_Target_Capacity(
                        candidate_sci_numbers,
                        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.capacity,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_high_water,
                        ClusteredNeighborProviderInternal::Workspace(layout)
                            .candidate_leaf_onepass_record_capacity,
                        false);
                Reserve_Candidate_Leaf_Onepass_Scratch(
                    layout, fused_record_capacity, false);
                Reserve_Device_Buffer(
                    2, &ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_cursor);
                deviceMemset(
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_cursor.data, 0,
                    sizeof(int) * 2);
                deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout)
                                 .candidate_leaf_queue2_task_work_cursor.data,
                             0, sizeof(int));
                {
                    Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Fused(
                        kFixedShiftCandidateLeafQueue2DeviceBlocks,
                        kClusteredBuilderBlockSize, candidate_sci_numbers,
                        candidate_sci_supercluster_ids,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell,
                        build_cutoff, ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.prefixes),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.childOffsets),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.parents),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.internalToLeaf),
                        candidate_shift_ids, false,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_work_cursor
                            .data,
                        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_leaf_counts
                            .data,
                        fused_record_capacity,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_cursor.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_cursor.data +
                            1,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_sci_ids.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_ranks.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_leaf_ids.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes
                            .data);
                }
                int h_fused_cursor[2] = {0, 0};
                deviceMemcpy(
                    h_fused_cursor,
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_cursor.data,
                    sizeof(int) * 2, deviceMemcpyDeviceToHost);
                if (h_fused_cursor[1] == 0 &&
                    h_fused_cursor[0] <= fused_record_capacity)
                {
                    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers =
                        ExclusiveScanCounts(
                            layout, candidate_sci_numbers,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data);
                    candidate_leaf_numbers_by_task = ExclusiveScanCounts(
                        layout, h_queue2_tasks,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_leaf_counts
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout)
                            .candidate_leaf_queue2_task_leaf_offsets.data);
                    if (ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers ==
                            h_fused_cursor[0] &&
                        candidate_leaf_numbers_by_task == h_fused_cursor[0])
                    {
                        candidate_leaf_queue2_counts_ready = true;
                        candidate_leaf_queue2_fused_used = true;
                        ClusteredNeighborProviderInternal::Workspace(layout)
                            .candidate_leaf_onepass_high_water = IntMax(
                            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_high_water,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers);
                        if (ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers > 0)
                        {
                            Reserve_Device_Buffer(
                                ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers,
                                &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids);
                            {
                                Launch_Device_Kernel(
                                    Scatter_Candidate_Leaves_From_Queue2_Task_Records,
                                    (ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers +
                                     CONTROLLER::device_max_thread - 1) /
                                        CONTROLLER::device_max_thread,
                                    CONTROLLER::device_max_thread, 0, NULL,
                                    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers,
                                    ClusteredNeighborProviderInternal::Workspace(layout)
                                        .candidate_leaf_queue2_task_leaf_offsets
                                        .data,
                                    ClusteredNeighborProviderInternal::Workspace(layout)
                                        .candidate_leaf_onepass_sci_ids.data,
                                    ClusteredNeighborProviderInternal::Workspace(layout)
                                        .candidate_leaf_onepass_ranks.data,
                                    ClusteredNeighborProviderInternal::Workspace(layout)
                                        .candidate_leaf_onepass_leaf_ids.data,
                                    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data);
                            }
                            candidate_leaf_queue2_ids_materialized = true;
                        }
                    }
                }
                else
                {
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_overflow_count +=
                        1;
                    const int required_capacity =
                        h_fused_cursor[0] > 0 ? h_fused_cursor[0]
                                              : fused_record_capacity + 1;
                    ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_high_water =
                        IntMax(
                            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_onepass_high_water,
                            required_capacity);
                }
                if (!candidate_leaf_queue2_fused_used)
                {
                    deviceMemset(ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data, 0,
                                 sizeof(int) * candidate_sci_numbers);
                    if (h_queue2_tasks > 0)
                    {
                        deviceMemset(
                            ClusteredNeighborProviderInternal::Workspace(layout)
                                .candidate_leaf_queue2_task_leaf_counts.data,
                            0, sizeof(int) * h_queue2_tasks);
                    }
                }
            }
            if (!candidate_leaf_queue2_fused_used)
            {
                deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout)
                                 .candidate_leaf_queue2_task_work_cursor.data,
                             0, sizeof(int));
                {
                    Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Count(
                        kFixedShiftCandidateLeafQueue2DeviceBlocks,
                        kClusteredBuilderBlockSize, candidate_sci_numbers,
                        candidate_sci_supercluster_ids,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell,
                        build_cutoff, ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
                        ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.prefixes),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.childOffsets),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.parents),
                        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state
                                           ->octree.internalToLeaf),
                        candidate_shift_ids, false,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_capacity,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_work_cursor
                            .data,
                        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_leaf_counts
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids
                            .data,
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes
                            .data);
                }
                ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers = ExclusiveScanCounts(
                    layout, candidate_sci_numbers,
                    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data);
                candidate_leaf_numbers_by_task =
                    h_queue2_tasks > 0
                        ? ExclusiveScanCounts(
                              layout, h_queue2_tasks,
                              ClusteredNeighborProviderInternal::Workspace(layout)
                                  .candidate_leaf_queue2_task_leaf_counts.data,
                              ClusteredNeighborProviderInternal::Workspace(layout)
                                  .candidate_leaf_queue2_task_leaf_offsets.data)
                        : 0;
                candidate_leaf_queue2_counts_ready = true;
                if (ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers > 0)
                {
                    Reserve_Device_Buffer(
                        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers,
                        &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids);
                    deviceMemset(
                        ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_work_cursor
                            .data,
                        0, sizeof(int));
                    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout)
                                     .candidate_leaf_queue2_task_overflow.data,
                                 0, sizeof(int));
                    {
                        Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Emit(
                            kFixedShiftCandidateLeafQueue2DeviceBlocks,
                            kClusteredBuilderBlockSize, candidate_sci_numbers,
                            candidate_sci_supercluster_ids,
                            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell,
                            build_cutoff, ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                            cstone::rawPtr(
                                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree
                                    .prefixes),
                            cstone::rawPtr(
                                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree
                                    .childOffsets),
                            cstone::rawPtr(
                                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree
                                    .parents),
                            cstone::rawPtr(
                                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree
                                    .internalToLeaf),
                            candidate_shift_ids, false,
                            ClusteredNeighborProviderInternal::Workspace(layout)
                                .candidate_leaf_queue2_task_capacity,
                            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_counter
                                .data,
                            ClusteredNeighborProviderInternal::Workspace(layout)
                                .candidate_leaf_queue2_task_work_cursor.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
                            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data,
                            ClusteredNeighborProviderInternal::Workspace(layout)
                                .candidate_leaf_queue2_task_leaf_offsets.data,
                            ClusteredNeighborProviderInternal::Workspace(layout)
                                .candidate_leaf_queue2_task_overflow.data,
                            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_sci_ids
                                .data,
                            ClusteredNeighborProviderInternal::Workspace(layout).candidate_leaf_queue2_task_nodes
                                .data);
                    }
                    candidate_leaf_queue2_ids_materialized = true;
                }
            }
        }
    }
#endif
    if (!candidate_leaf_queue2_ids_materialized &&
        !candidate_leaf_queue2_counts_ready)
    {
#ifdef USE_CPU
        Launch_Device_Kernel(
            Count_Supercluster_Candidate_Leaves_Fixed_Shift, 1, 1, 0, NULL,
            candidate_sci_numbers, candidate_sci_supercluster_ids,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell, build_cutoff,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.prefixes),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.childOffsets),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.parents),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, true,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data);
#else
        const int candidate_leaf_groups_per_block =
            kClusteredBuilderBlockSize / kFixedShiftCandidateLeafSubgroupSize;
        Launch_Device_Kernel(
            Count_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup,
            (candidate_sci_numbers + candidate_leaf_groups_per_block - 1) /
                candidate_leaf_groups_per_block,
            kClusteredBuilderBlockSize, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell, build_cutoff,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.prefixes),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.childOffsets),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.parents),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data);
#endif
    }
    if (!candidate_leaf_queue2_ids_materialized &&
        !candidate_leaf_queue2_counts_ready)
    {
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers =
            ExclusiveScanCounts(layout, candidate_sci_numbers,
                                  ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                                  ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data);
    }
    if (ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers <= 0)
    {
        CommitBuildCache(layout, commit_cache_crd, cutoff);
        return {};
    }

    if (!candidate_leaf_queue2_ids_materialized)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers,
                              &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids);
    }
    if (!candidate_leaf_queue2_ids_materialized)
    {
#ifdef USE_CPU
        Launch_Device_Kernel(
            Fill_Supercluster_Candidate_Leaves_Fixed_Shift, 1, 1, 0, NULL,
            candidate_sci_numbers, candidate_sci_supercluster_ids,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell, build_cutoff,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.prefixes),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.childOffsets),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.parents),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
            true, ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data);
#else
        const int candidate_leaf_groups_per_block =
            kClusteredBuilderBlockSize / kFixedShiftCandidateLeafSubgroupSize;
        Launch_Device_Kernel(
            Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup,
            (candidate_sci_numbers + candidate_leaf_groups_per_block - 1) /
                candidate_leaf_groups_per_block,
            kClusteredBuilderBlockSize, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, cell, build_cutoff,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.prefixes),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.childOffsets),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.parents),
            cstone::rawPtr(
                ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data);
#endif
    }
    const int candidate_leaf_mask_blocks =
        (candidate_sci_numbers +
         (kClusteredBuilderBlockSize / kClusteredBuilderWarpSize) - 1) /
        (kClusteredBuilderBlockSize / kClusteredBuilderWarpSize);
    Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers *
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_cluster_stride,
        &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_reach_masks);
    Launch_Device_Kernel(
        Build_Fixed_Shift_Candidate_Leaf_Masks, candidate_leaf_mask_blocks,
        kClusteredBuilderBlockSize, 0, NULL, candidate_sci_numbers,
        ClusteredNeighborProviderInternal::Config(layout).clusters_per_supercluster, build_cutoff, cell,
        candidate_sci_supercluster_ids,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data,
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data, candidate_shift_ids,
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_cluster_stride,
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_reach_masks.data);

    ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers = candidate_sci_numbers;
    return {clustered_neighbor_builder_internal::BuildPreparationResult::ready,
            candidate_leaf_queue2_ids_materialized ||
                candidate_leaf_queue2_counts_ready};
}

}  // namespace clustered_neighbor_builder_internal
