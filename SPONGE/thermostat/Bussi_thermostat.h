#ifndef BUSSI_THERMOSTAT_H
#define BUSSI_THERMOSTAT_H

#include <sstream>

#include "../common.h"
#include "../control.h"
#include "../utils/h5md/h5_structural_state.hpp"

// 用于记录与计算Bussi CVR控温相关的信息
struct BUSSI_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260227;

    float tauT;                // 弛豫时间（ps）
    float dt;                  // 步长（ps）
    float target_temperature;  // 目标温度
    float lambda;              // 速度缩放系数

    std::default_random_engine e;
    std::normal_distribution<float> normal01;

    // 初始化
    void Initial(CONTROLLER* controller, float target_temperature,
                 const char* module_name = NULL);

    // 根据当前温度计算Bussi精确CVR缩放系数
    void Record_Temperature(float temperature, int freedom);

    // 按lambda缩放速度
    void Scale_Velocity(int atom_numbers, VECTOR* vel);
    void Set_Target_Temperature(float target_temperature_new);
    bool Export_H5_Restart_State(SpongeH5MD::RestartDynamicState* state,
                                 std::string* error_message) const;
    bool Apply_H5_Restart_State(const SpongeH5MD::RestartDynamicState& state,
                                std::string* error_message);
};

inline bool BUSSI_THERMOSTAT_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartDynamicState* state, std::string* error_message) const
{
    if (state == nullptr)
    {
        if (error_message != nullptr)
        {
            *error_message = "Bussi H5 restart state output pointer is null";
        }
        return false;
    }
    if (!is_initialized)
    {
        return true;
    }
    std::ostringstream rng;
    rng << e;
    const std::string module = "bussi_thermostat";
    state->rng_state_text[module] = rng.str();
    state->thermostat_text_states[module]["rng_engine"] =
        "std::default_random_engine";
    state->thermostat_float_states[module]["lambda"] = {lambda};
    return true;
}

inline bool BUSSI_THERMOSTAT_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartDynamicState& state, std::string* error_message)
{
    const std::string module = "bussi_thermostat";
    const auto rng_state = state.rng_state_text.find(module);
    if (rng_state == state.rng_state_text.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Current run uses bussi_thermostat, but restart does not "
                "contain Bussi thermostat RNG state";
        }
        return false;
    }
    if (!is_initialized)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Restart contains Bussi thermostat RNG state, but the "
                "bussi_thermostat module is not initialized";
        }
        return false;
    }
    std::istringstream rng(rng_state->second);
    rng >> e;
    if (rng.fail())
    {
        if (error_message != nullptr)
        {
            *error_message = "Failed to parse Bussi thermostat RNG state";
        }
        return false;
    }
    const auto module_floats = state.thermostat_float_states.find(module);
    if (module_floats == state.thermostat_float_states.end())
    {
        if (error_message != nullptr)
        {
            *error_message = "Bussi thermostat restart state is missing lambda";
        }
        return false;
    }
    const auto lambda_state = module_floats->second.find("lambda");
    if (lambda_state == module_floats->second.end() ||
        lambda_state->second.size() != 1)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Bussi thermostat restart state is missing a scalar lambda";
        }
        return false;
    }
    lambda = lambda_state->second[0];
    return true;
}

#endif
