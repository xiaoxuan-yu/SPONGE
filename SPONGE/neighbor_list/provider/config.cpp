#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "../../control.h"

ClusteredBuildConfig ResolveClusteredNeighborConfig(
    CONTROLLER* controller, const char* module_name)
{
    ClusteredBuildConfig config;
    if (controller->Command_Exist("LJ", "cluster_size"))
    {
        controller->Check_Int("LJ", "cluster_size", module_name);
        config.cluster_size = atoi(controller->Command("LJ", "cluster_size"));
    }
    if (controller->Command_Exist("LJ", "super_cluster_clusters"))
    {
        controller->Check_Int("LJ", "super_cluster_clusters", module_name);
        config.clusters_per_supercluster =
            atoi(controller->Command("LJ", "super_cluster_clusters"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_max_depth"))
    {
        controller->Check_Int("LJ", "cornerstone_max_depth", module_name);
        config.cornerstone_max_depth =
            atoi(controller->Command("LJ", "cornerstone_max_depth"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_leaf_size"))
    {
        controller->Check_Int("LJ", "cornerstone_leaf_size", module_name);
        config.cornerstone_leaf_size =
            atoi(controller->Command("LJ", "cornerstone_leaf_size"));
    }
    if (controller->Command_Exist("LJ", "cpu_simd"))
    {
#ifndef USE_CPU
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "ResolveLJClusteredBuildConfig",
            "Reason:\n\t clustered direct LJ on GPU requires a "
            "direct gmxpacked payload build; remove the deprecated "
            "LJ.cpu_simd setting.\n");
#endif
    }
    if (controller->Command_Exist("LJ", "clustered_rebuild_skin"))
    {
        controller->Check_Float("LJ", "clustered_rebuild_skin", module_name);
        config.rebuild_skin =
            atof(controller->Command("LJ", "clustered_rebuild_skin"));
    }
    if (controller->Command_Exist("neighbor_list", "skin_permit"))
    {
        controller->Check_Float("neighbor_list", "skin_permit", module_name);
        config.skin_permit =
            atof(controller->Command("neighbor_list", "skin_permit"));
    }
    if (controller->Command_Exist("neighbor_list", "refresh_interval"))
    {
        controller->Check_Int("neighbor_list", "refresh_interval", module_name);
        config.refresh_interval =
            atoi(controller->Command("neighbor_list", "refresh_interval"));
    }

    config.cluster_size = std::max(1, config.cluster_size);
    config.clusters_per_supercluster =
        std::max(1, config.clusters_per_supercluster);
    config.cornerstone_max_depth =
        std::max(1, std::min(21, config.cornerstone_max_depth));
    config.cornerstone_leaf_size = std::max(1, config.cornerstone_leaf_size);
    config.rebuild_skin = fmaxf(0.0f, config.rebuild_skin);
    config.skin_permit = fmaxf(0.0f, config.skin_permit);

    if (config.cluster_size != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only cluster_size=8; "
            "override %d is ignored.\n",
            config.cluster_size);
        config.cluster_size = 8;
    }
    if (config.clusters_per_supercluster != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only "
            "super_cluster_clusters=8; override %d is ignored.\n",
            config.clusters_per_supercluster);
        config.clusters_per_supercluster = 8;
    }
    controller->printf(
        "    direct_kernel: clustered (cluster_size=%d "
        "super_cluster_clusters=%d depth=%d leaf_size=%d reuse_skin=%.2f "
        "skin_permit=%.2f refresh_interval=%d)\n",
        config.cluster_size, config.clusters_per_supercluster,
        config.cornerstone_max_depth, config.cornerstone_leaf_size,
        config.rebuild_skin, config.skin_permit, config.refresh_interval);
    return config;
}
