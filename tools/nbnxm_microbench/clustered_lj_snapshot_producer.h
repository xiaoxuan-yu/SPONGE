#pragma once

#include <string>

namespace nbnxm_microbench
{

void WriteCurrentClusteredLJSnapshots(const std::string& prefix,
                                      bool write_full_output);

}  // namespace nbnxm_microbench
