#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sponge
{

struct RuntimeStateAtom
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct RuntimeState
{
    int atom_count = 0;
    int step = 0;
    int step_limit = 0;
    double start_time_ps = 0.0;
    double current_time_ps = 0.0;
    std::array<float, 3> box_length = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> box_angle = {0.0f, 0.0f, 0.0f};
    std::vector<RuntimeStateAtom> coordinates;
    std::vector<RuntimeStateAtom> velocities;
    std::vector<RuntimeStateAtom> local_accelerations;
    std::vector<float> nhc_coordinates;
    std::vector<float> nhc_velocities;
    std::vector<RuntimeStateAtom> settle_last_pair_ab;
    std::vector<RuntimeStateAtom> settle_last_triangle_ba;
    std::vector<RuntimeStateAtom> settle_last_triangle_ca;
    std::vector<RuntimeStateAtom> shake_last_pair_dr;
    std::array<float, 6> pressure_barostat_g = {0.0f, 0.0f, 0.0f,
                                                0.0f, 0.0f, 0.0f};
    float pressure_barostat_v0 = 0.0f;
    std::string pressure_barostat_rng_state;
    std::string pressure_barostat_distribution_state;
    std::array<int, 3> mc_barostat_total_count = {0, 0, 0};
    std::array<int, 3> mc_barostat_accept_count = {0, 0, 0};
    std::array<float, 3> mc_barostat_accept_rate = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> mc_barostat_delta_box_length_max = {0.0f, 0.0f, 0.0f};
    std::string mc_barostat_rng_state;
    std::vector<std::uint8_t> middle_langevin_rng_state;
    std::vector<std::uint8_t> andersen_rng_state;
    std::string bussi_rng_state;
    std::string bussi_distribution_state;
    bool has_local_accelerations = false;
    bool has_nhc_state = false;
    bool has_settle_state = false;
    bool has_shake_state = false;
    bool has_pressure_barostat_state = false;
    bool has_mc_barostat_state = false;
    bool has_middle_langevin_rng_state = false;
    bool has_andersen_rng_state = false;
    bool has_bussi_rng_state = false;
    bool valid = false;
};

struct WorkerExchangeObservable
{
    int step = 0;
    double time_ps = 0.0;
    float total_potential = 0.0f;
    float effective_potential = 0.0f;
    float temperature = 0.0f;
    float target_temperature = 0.0f;
    float pressure = 0.0f;
    float target_pressure = 0.0f;
    float volume = 0.0f;
};

struct SchedulerSnapshot
{
    int next_step = 0;
    int last_completed_step = -1;
    int step_limit = 0;
    double current_time_ps = 0.0;
    double dt_ps = 0.0;
    float temperature = 0.0f;
    float target_temperature = 0.0f;
    float pressure = 0.0f;
    float target_pressure = 0.0f;
    float total_potential = 0.0f;
    float effective_potential = 0.0f;
    std::array<float, 3> box_length = {0.0f, 0.0f, 0.0f};
    bool initialized = false;
    bool finished = false;
};

/*
 * A lightweight single-replica execution wrapper for SPONGE.
 *
 * This interface is intentionally process-local and single-instance oriented:
 * one SpongeScheduler owns one SPONGE runtime in the current process.
 * Higher-level orchestration should launch multiple processes or reuse this
 * interface sequentially when building a replica scheduler.
 */
class SpongeScheduler
{
   public:
    SpongeScheduler() = default;
    ~SpongeScheduler();

    void InitializeFromArgv(int argc, char** argv);
    void InitializeFromArgs(const std::vector<std::string>& args);

    void RunSingleStep(bool emit_output = true);
    void RunSteps(int steps, bool emit_output = true);
    void RunToEnd(bool emit_output = true);

    SchedulerSnapshot Snapshot() const;
    RuntimeState ExportRuntimeState();
    void ImportRuntimeState(const RuntimeState& state);
    WorkerExchangeObservable CollectExchangeObservables() const;
    void EnsureForeignStateProbeSafe() const;
    void ScaleVelocities(float factor);
    void InvalidateNeighborList(bool rebuild_dd = true);

    bool IsInitialized() const;
    bool IsFinished() const;

    void Finalize();

   private:
    void InitializeFromOwnedArgs();
    void EnsureInitialized(const char* caller) const;
    void RebuildArgvCache();

    bool initialized_ = false;
    bool finalized_ = false;
    std::vector<std::string> owned_args_;
    std::vector<char*> argv_cache_;
};

}  // namespace sponge
