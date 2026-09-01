#pragma once

#include "../provider/provider.h"

struct ClusteredNeighborProvider::BuildPayloadInput
{
    const VECTOR* crd = NULL;
    const VECTOR* commit_cache_crd = NULL;
    LTMatrix3 cell = {};
    LTMatrix3 rcell = {};
    float cutoff = 0.0f;
    float build_cutoff = 0.0f;
    bool candidate_leaf_queue2_payload_ready = false;
};
