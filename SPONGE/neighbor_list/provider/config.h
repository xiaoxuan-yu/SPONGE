#pragma once

struct CONTROLLER;

struct ClusteredBuildConfig
{
    int cluster_size = 8;
    int clusters_per_supercluster = 8;
    int cornerstone_max_depth = 6;
    int cornerstone_leaf_size = 32;
    float rebuild_skin = 10.0f;
    float skin_permit = 0.5f;
    int refresh_interval = 0;
};

ClusteredBuildConfig ResolveClusteredNeighborConfig(
    CONTROLLER* controller, const char* module_name);
