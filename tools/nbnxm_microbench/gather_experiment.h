#pragma once

#include "nbnxm_microbench_snapshot.h"

namespace nbnxm_microbench
{

int RunGatherExperiment(const SpongeGmxpackedForceOnlySnapshot& snapshot,
                        int warmup, int iters, int block_size);

}  // namespace nbnxm_microbench
