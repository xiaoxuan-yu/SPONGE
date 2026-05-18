#pragma once

#include <string>

#include "core/manager.h"

namespace sponge::manager
{

struct ManagerExecutionConfig
{
    Manager::ManagerConfig manager;
    int epochs = 1;
    bool emit_output = false;
    std::string remd_mode;
    int exchange_round = 0;
};

ManagerExecutionConfig LoadManagerExecutionConfigFromToml(
    const std::string& config_path, const std::string& manager_executable_path);

}  // namespace sponge::manager
