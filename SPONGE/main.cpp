#include "main.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

#define SUBPACKAGE_HINT \
    "SPONGE, for general-purpose molecular dynamics simulations"
#define THERMOSTAT_IS(name)                              \
    (md_info.mode >= md_info.NVT &&                      \
     (controller.Command_Choice("thermostat", (name)) || \
      controller.Command_Choice("thermostat_mode", (name))))
#define BAROSTAT_IS(name)                              \
    (md_info.mode == md_info.NPT &&                    \
     (controller.Command_Choice("barostat", (name)) || \
      controller.Command_Choice("barostat_mode", (name))))

CONTROLLER controller;
Xponge::System Xponge::system;
MD_INFORMATION md_info;
DOMAIN_INFORMATION dd;
MIDDLE_Langevin_INFORMATION middle_langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
BUSSI_THERMOSTAT_INFORMATION bussi_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
PRESSURE_BASED_BAROSTAT_INFORMATION press_baro;
MC_BAROSTAT_INFORMATION mc_baro;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
LJ_SOFT_CORE lj_soft;
SOLVENT_LENNARD_JONES solvent_lj;
Particle_Mesh pm;
ANGLE angle;
UREY_BRADLEY urey_bradley;
BOND bond;
CMAP cmap;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
STEER_CV steer_cv;
RESTRAIN_CV restrain_cv;
META meta;
LISTED_FORCES listed_forces;
PAIRWISE_FORCE pairwise_force;
LJ_CLUSTERED_DIRECT_CACHE* pairwise_clustered_cache = NULL;
HARD_WALL hard_wall;
SOFT_WALLS soft_walls;
LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;
SELECTIVE_INTERACTION selective_interaction;
DIHEDRAL sits_dihedral;
NON_BOND_14 sits_nb14;
CMAP sits_cmap;
STILLINGER_WEBER_INFORMATION sw;
EDIP_INFORMATION edip;
EAM_INFORMATION eam;
TERSOFF_INFORMATION tersoff;
REAXFF reaxff;
QUANTUM_CHEMISTRY qc;
SPONGE_PLUGIN plugin;

deviceStream_t main_stream;

namespace
{

sponge::RuntimeStateAtom Make_Runtime_State_Atom(const VECTOR& value)
{
    return {value.x, value.y, value.z};
}

VECTOR Make_Vector(const sponge::RuntimeStateAtom& value)
{
    return {value.x, value.y, value.z};
}

bool Main_Finite_Lifecycle_Probe_Env_Enabled(const char* name)
{
    const char* enabled = std::getenv(name);
    return enabled != NULL && enabled[0] != '\0' &&
           !(enabled[0] == '0' && enabled[1] == '\0');
}

bool Main_Finite_Lifecycle_Probe_Enabled()
{
    static const bool enabled = Main_Finite_Lifecycle_Probe_Env_Enabled(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE");
    return enabled;
}

bool Main_Finite_Lifecycle_Probe_Abort_Enabled()
{
    static const bool enabled = Main_Finite_Lifecycle_Probe_Env_Enabled(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_ABORT");
    return enabled;
}

bool Main_Finite_Async_Lifecycle_Probe_Enabled()
{
    static const bool enabled = Main_Finite_Lifecycle_Probe_Env_Enabled(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE");
    return enabled;
}

bool Main_Finite_Async_Lifecycle_Probe_Stop_Enabled()
{
    static const bool enabled = Main_Finite_Lifecycle_Probe_Env_Enabled(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE_STOP");
    return enabled;
}

int Main_Finite_Lifecycle_Probe_Begin_Step()
{
    static const int begin_step = [] {
        const char* value = std::getenv(
            "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_BEGIN");
        return value == NULL ? 0 : atoi(value);
    }();
    return begin_step;
}

int Main_Finite_Lifecycle_Probe_End_Step()
{
    static const int end_step = [] {
        const char* value = std::getenv(
            "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_END");
        return value == NULL ? -1 : atoi(value);
    }();
    return end_step;
}

int Main_Finite_Lifecycle_Probe_Interval()
{
    static const int interval = [] {
        const char* value = std::getenv(
            "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_INTERVAL");
        const int parsed = value == NULL ? 1 : atoi(value);
        return parsed <= 0 ? 1 : parsed;
    }();
    return interval;
}

const char* Main_Finite_Lifecycle_Probe_Site_Filter()
{
    static const char* site = std::getenv(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_SITE");
    return site != NULL && site[0] != '\0' ? site : NULL;
}

const char* const* Main_Finite_Probe_Site_Names(int* count)
{
    static const char* sites[] = {
        "run_step_begin",
        "after_sync_dynamic_targets",
        "after_calculate_force",
        "after_iteration",
        "after_print",
        "after_step_increment",
        "force_entry",
        "after_md_reset_force",
        "after_qc_scf",
        "after_dd_reset_force",
        "after_qc_gradient",
        "after_update_ghost",
        "after_force_neighbor_update",
        "after_reaxff_force",
        "after_selective_update",
        "after_vatom_force_redistribute",
        "before_scale_force_dynamic_dt",
        "force_exit",
        "refresh_entry",
        "refresh_after_get_atoms",
        "refresh_after_get_ghost",
        "refresh_after_ordered_layout",
        "refresh_after_get_excluded",
        "refresh_exit",
        "iteration_entry",
        "iteration_after_ek_temperature",
        "iteration_after_potential",
        "iteration_after_mc_barostat",
        "iteration_before_remember_last_coordinates",
        "iteration_after_remember_last_coordinates",
        "iteration_before_integrator",
        "iteration_after_nve",
        "iteration_after_minimization",
        "iteration_after_middle_langevin",
        "iteration_after_berendsen",
        "iteration_after_bussi",
        "iteration_after_andersen",
        "iteration_after_andersen_nve",
        "iteration_after_nhc",
        "iteration_before_settle",
        "iteration_after_settle",
        "iteration_before_shake",
        "iteration_after_shake",
        "iteration_before_hard_wall",
        "iteration_after_hard_wall",
        "iteration_after_pressure",
        "iteration_after_rerun",
        "iteration_after_rerun_box_change",
        "iteration_after_rerun_global_to_dd",
        "iteration_after_vatom_coord_refresh",
        "iteration_after_exchange_particles",
        "iteration_after_refresh_local_state",
        "iteration_after_single_rank_neighbor_update",
        "iteration_after_pm_get_atoms",
        "iteration_exit",
    };
    *count = static_cast<int>(sizeof(sites) / sizeof(sites[0]));
    return sites;
}

int Main_Finite_Probe_Site_Id(const char* site)
{
    int count = 0;
    const char* const* sites = Main_Finite_Probe_Site_Names(&count);
    for (int i = 0; i < count; i++)
    {
        if (std::strcmp(site, sites[i]) == 0)
        {
            return i;
        }
    }
    return -1;
}

const char* Main_Finite_Probe_Site_Name(int site_id)
{
    int count = 0;
    const char* const* sites = Main_Finite_Probe_Site_Names(&count);
    return site_id >= 0 && site_id < count ? sites[site_id] : "unknown";
}

const char* Main_Finite_Probe_Array_Name(int array_id)
{
    static const char* arrays[] = {
        "dd.crd",      "dd.vel",      "dd.frc",    "dd.acc",
        "md_info.crd", "md_info.vel", "md_info.frc"};
    const int count = static_cast<int>(sizeof(arrays) / sizeof(arrays[0]));
    return array_id >= 0 && array_id < count ? arrays[array_id] : "unknown";
}

int Main_Finite_Probe_Array_Count()
{
    return 7;
}

const char* Main_Finite_Probe_Array_Filter()
{
    static const char* array = std::getenv(
        "SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE_ARRAY");
    return array != NULL && array[0] != '\0' ? array : NULL;
}

bool Main_Finite_Probe_Array_Filter_Allows(int array_id)
{
    const char* array_filter = Main_Finite_Probe_Array_Filter();
    return array_filter == NULL ||
           std::strcmp(array_filter, Main_Finite_Probe_Array_Name(array_id)) ==
               0;
}

bool Main_Vector_Is_Finite(const VECTOR& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

struct Main_Finite_Async_Probe_Record
{
    int found;
    int step;
    int site_id;
    int array_id;
    int local_atom;
    int global_atom;
    float x;
    float y;
    float z;
};

Main_Finite_Async_Probe_Record* main_finite_async_probe_record = NULL;
int main_finite_async_probe_record_count = 0;

void Main_Finite_Async_Lifecycle_Probe_Report();

__device__ __forceinline__ bool Main_Device_Vector_Is_Finite(VECTOR value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__global__ void Main_Finite_Async_Scan_Vector_Array(
    int step, int site_id, int array_id, const VECTOR* values, int atom_count,
    const int* local_to_global, Main_Finite_Async_Probe_Record* record)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (record == NULL || values == NULL || atom >= atom_count ||
        record->found != 0)
    {
        return;
    }
    const VECTOR value = values[atom];
    if (Main_Device_Vector_Is_Finite(value))
    {
        return;
    }
    if (atomicCAS(&record->found, 0, 1) == 0)
    {
        record->step = step;
        record->site_id = site_id;
        record->array_id = array_id;
        record->local_atom = atom;
        record->global_atom = local_to_global != NULL ? local_to_global[atom]
                                                      : atom;
        record->x = value.x;
        record->y = value.y;
        record->z = value.z;
    }
}

void Main_Finite_Async_Lifecycle_Probe_Ensure_Record()
{
    if (!Main_Finite_Async_Lifecycle_Probe_Enabled() ||
        main_finite_async_probe_record != NULL)
    {
        return;
    }
    int site_count = 0;
    Main_Finite_Probe_Site_Names(&site_count);
    main_finite_async_probe_record_count =
        site_count * Main_Finite_Probe_Array_Count();
    Device_Malloc_Safely((void**)&main_finite_async_probe_record,
                         sizeof(Main_Finite_Async_Probe_Record) *
                             main_finite_async_probe_record_count);
    deviceMemset(main_finite_async_probe_record, 0,
                 sizeof(Main_Finite_Async_Probe_Record) *
                     main_finite_async_probe_record_count);
}

bool Main_Finite_Lifecycle_Probe_Filter_Allows(const char* site)
{
    const int step = md_info.sys.steps;
    const int begin_step = Main_Finite_Lifecycle_Probe_Begin_Step();
    const int end_step = Main_Finite_Lifecycle_Probe_End_Step();
    const int interval = Main_Finite_Lifecycle_Probe_Interval();
    const char* site_filter = Main_Finite_Lifecycle_Probe_Site_Filter();
    return step >= begin_step && (end_step < 0 || step <= end_step) &&
           ((step - begin_step) % interval) == 0 &&
           (site_filter == NULL || std::strcmp(site_filter, site) == 0);
}

void Main_Finite_Async_Scan_One_Array(const char* site, int site_id,
                                      int array_id, const VECTOR* values,
                                      int atom_count,
                                      const int* local_to_global)
{
    if (values == NULL || atom_count <= 0 ||
        !Main_Finite_Probe_Array_Filter_Allows(array_id) || site_id < 0 ||
        array_id < 0 || array_id >= Main_Finite_Probe_Array_Count())
    {
        return;
    }
    const int threads = 256;
    const int blocks = (atom_count + threads - 1) / threads;
    const int record_id = site_id * Main_Finite_Probe_Array_Count() + array_id;
    if (record_id < 0 || record_id >= main_finite_async_probe_record_count)
    {
        return;
    }
    Launch_Device_Kernel(Main_Finite_Async_Scan_Vector_Array, blocks, threads,
                         0, NULL, md_info.sys.steps, site_id, array_id, values,
                         atom_count, local_to_global,
                         main_finite_async_probe_record + record_id);
}

void Main_Finite_Async_Lifecycle_Probe(const char* site)
{
    if (!Main_Finite_Async_Lifecycle_Probe_Enabled())
    {
        return;
    }
    Main_Finite_Async_Lifecycle_Probe_Ensure_Record();
    if (main_finite_async_probe_record == NULL ||
        !Main_Finite_Lifecycle_Probe_Filter_Allows(site))
    {
        return;
    }

    const int site_id = Main_Finite_Probe_Site_Id(site);
    const int* local_to_global =
        CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size ? dd.atom_local : NULL;
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size && dd.atom_numbers > 0)
    {
        Main_Finite_Async_Scan_One_Array(site, site_id, 0, dd.crd,
                                         dd.atom_numbers, local_to_global);
        Main_Finite_Async_Scan_One_Array(site, site_id, 1, dd.vel,
                                         dd.atom_numbers, local_to_global);
        Main_Finite_Async_Scan_One_Array(site, site_id, 2, dd.frc,
                                         dd.atom_numbers, local_to_global);
        Main_Finite_Async_Scan_One_Array(site, site_id, 3, dd.acc,
                                         dd.atom_numbers, local_to_global);
    }
    Main_Finite_Async_Scan_One_Array(site, site_id, 4, md_info.crd,
                                     md_info.atom_numbers, NULL);
    Main_Finite_Async_Scan_One_Array(site, site_id, 5, md_info.vel,
                                     md_info.atom_numbers, NULL);
    Main_Finite_Async_Scan_One_Array(site, site_id, 6, md_info.frc,
                                     md_info.atom_numbers, NULL);
    if (Main_Finite_Async_Lifecycle_Probe_Stop_Enabled())
    {
        Main_Finite_Async_Lifecycle_Probe_Report();
        std::fflush(stderr);
        std::exit(0);
    }
}

void Main_Finite_Async_Lifecycle_Probe_Report()
{
    if (!Main_Finite_Async_Lifecycle_Probe_Enabled() ||
        main_finite_async_probe_record == NULL)
    {
        return;
    }
    std::vector<Main_Finite_Async_Probe_Record> records(
        main_finite_async_probe_record_count);
    deviceMemcpy(records.data(), main_finite_async_probe_record,
                 sizeof(Main_Finite_Async_Probe_Record) * records.size(),
                 deviceMemcpyDeviceToHost);
    bool any_found = false;
    for (const Main_Finite_Async_Probe_Record& record : records)
    {
        if (record.found == 0)
        {
            continue;
        }
        any_found = true;
        std::fprintf(
            stderr,
            "SPONGE_FINITE_ASYNC_LIFECYCLE_PROBE step=%d site=%s site_id=%d "
            "array=%s array_id=%d first_local=%d first_global=%d "
            "value=(%.9g,%.9g,%.9g)\n",
            record.step, Main_Finite_Probe_Site_Name(record.site_id),
            record.site_id, Main_Finite_Probe_Array_Name(record.array_id),
            record.array_id, record.local_atom, record.global_atom, record.x,
            record.y, record.z);
    }
    if (!any_found)
    {
        std::fprintf(stderr,
                     "SPONGE_FINITE_ASYNC_LIFECYCLE_PROBE result=finite\n");
    }
}

struct Main_Finite_Array_Report
{
    int bad_atoms = 0;
    int first_local_atom = -1;
    int first_global_atom = -1;
    VECTOR first_value;
};

Main_Finite_Array_Report Main_Scan_Finite_Vector_Array(
    const char* site, const char* name, const VECTOR* device_values,
    int atom_count, const std::vector<int>* local_to_global)
{
    Main_Finite_Array_Report report;
    if (device_values == NULL || atom_count <= 0)
    {
        return report;
    }

    std::vector<VECTOR> values(atom_count);
    deviceMemcpy(values.data(), device_values, sizeof(VECTOR) * atom_count,
                 deviceMemcpyDeviceToHost);
    for (int i = 0; i < atom_count; i++)
    {
        if (Main_Vector_Is_Finite(values[i]))
        {
            continue;
        }
        if (report.bad_atoms == 0)
        {
            report.first_local_atom = i;
            report.first_global_atom =
                local_to_global != NULL ? (*local_to_global)[i] : i;
            report.first_value = values[i];
        }
        report.bad_atoms++;
    }

    if (report.bad_atoms > 0)
    {
        std::fprintf(
            stderr,
            "SPONGE_FINITE_LIFECYCLE_PROBE step=%d site=%s array=%s "
            "bad_atoms=%d first_local=%d first_global=%d "
            "value=(%.9g,%.9g,%.9g)\n",
            md_info.sys.steps, site, name, report.bad_atoms,
            report.first_local_atom, report.first_global_atom,
            report.first_value.x, report.first_value.y, report.first_value.z);
    }
    return report;
}

void Main_Finite_Lifecycle_Probe(const char* site)
{
    Main_Finite_Async_Lifecycle_Probe(site);

    static bool reported_first_bad = false;
    if (!Main_Finite_Lifecycle_Probe_Enabled() || reported_first_bad ||
        !Main_Finite_Lifecycle_Probe_Filter_Allows(site))
    {
        return;
    }

    bool any_bad = false;
    std::vector<int> dd_local_to_global;
    const bool has_dd_atoms =
        CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size && dd.atom_numbers > 0;
    if (has_dd_atoms && dd.atom_local != NULL)
    {
        dd_local_to_global.resize(dd.atom_numbers);
        deviceMemcpy(dd_local_to_global.data(), dd.atom_local,
                     sizeof(int) * dd.atom_numbers, deviceMemcpyDeviceToHost);
    }
    const std::vector<int>* local_to_global =
        dd_local_to_global.empty() ? NULL : &dd_local_to_global;

    if (has_dd_atoms)
    {
        any_bad |= Main_Scan_Finite_Vector_Array(
                       site, "dd.crd", dd.crd, dd.atom_numbers,
                       local_to_global)
                       .bad_atoms > 0;
        any_bad |= Main_Scan_Finite_Vector_Array(
                       site, "dd.vel", dd.vel, dd.atom_numbers,
                       local_to_global)
                       .bad_atoms > 0;
        any_bad |= Main_Scan_Finite_Vector_Array(
                       site, "dd.frc", dd.frc, dd.atom_numbers,
                       local_to_global)
                       .bad_atoms > 0;
        any_bad |= Main_Scan_Finite_Vector_Array(
                       site, "dd.acc", dd.acc, dd.atom_numbers,
                       local_to_global)
                       .bad_atoms > 0;
    }

    any_bad |= Main_Scan_Finite_Vector_Array(
                   site, "md_info.crd", md_info.crd, md_info.atom_numbers, NULL)
                   .bad_atoms > 0;
    any_bad |= Main_Scan_Finite_Vector_Array(
                   site, "md_info.vel", md_info.vel, md_info.atom_numbers, NULL)
                   .bad_atoms > 0;
    any_bad |= Main_Scan_Finite_Vector_Array(
                   site, "md_info.frc", md_info.frc, md_info.atom_numbers, NULL)
                   .bad_atoms > 0;

    if (any_bad)
    {
        reported_first_bad = true;
        std::fflush(stderr);
        if (Main_Finite_Lifecycle_Probe_Abort_Enabled())
        {
            std::abort();
        }
    }
}

std::vector<sponge::RuntimeStateAtom> Copy_Device_Vector_Array_To_Runtime_State(
    const VECTOR* device_pointer, std::size_t count)
{
    std::vector<sponge::RuntimeStateAtom> values(count);
    if (count == 0)
    {
        return values;
    }
    std::vector<VECTOR> host_values(count);
    deviceMemcpy(host_values.data(), device_pointer, sizeof(VECTOR) * count,
                 deviceMemcpyDeviceToHost);
    for (std::size_t i = 0; i < count; i++)
    {
        values[i] = Make_Runtime_State_Atom(host_values[i]);
    }
    return values;
}

void Copy_Runtime_State_Vector_Array_To_Device(
    const std::vector<sponge::RuntimeStateAtom>& values, VECTOR* device_pointer,
    std::size_t expected_count, const char* error_by, const char* blob_name)
{
    if (values.size() != expected_count)
    {
        std::string reason = "Reason:\n\tunexpected vector count for ";
        reason += blob_name;
        reason += "\n";
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand, error_by,
                                      reason.c_str());
    }
    if (expected_count == 0)
    {
        return;
    }
    std::vector<VECTOR> host_values(expected_count);
    for (std::size_t i = 0; i < expected_count; i++)
    {
        host_values[i] = Make_Vector(values[i]);
    }
    deviceMemcpy(device_pointer, host_values.data(),
                 sizeof(VECTOR) * expected_count, deviceMemcpyHostToDevice);
}

template <typename T>
std::vector<std::uint8_t> Copy_Device_Bytes_To_Host(T* device_pointer,
                                                    std::size_t count)
{
    std::vector<std::uint8_t> bytes(sizeof(T) * count);
    if (!bytes.empty())
    {
        deviceMemcpy(bytes.data(), device_pointer, bytes.size(),
                     deviceMemcpyDeviceToHost);
    }
    return bytes;
}

template <typename T>
void Copy_Host_Bytes_To_Device(const std::vector<std::uint8_t>& bytes,
                               T* device_pointer, std::size_t count,
                               const char* error_by, const char* blob_name)
{
    const std::size_t expected_size = sizeof(T) * count;
    if (bytes.size() != expected_size)
    {
        std::string reason = "Reason:\n\tunexpected byte count for ";
        reason += blob_name;
        reason += "\n";
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand, error_by,
                                      reason.c_str());
    }
    if (expected_size > 0)
    {
        deviceMemcpy(device_pointer, bytes.data(), expected_size,
                     deviceMemcpyHostToDevice);
    }
}

template <typename Engine>
std::string Serialize_Stream_State(const Engine& engine)
{
    std::ostringstream oss;
    oss << engine;
    return oss.str();
}

template <typename Engine>
void Deserialize_Stream_State(const std::string& serialized, Engine* engine,
                              const char* error_by, const char* state_name)
{
    if (engine == nullptr || serialized.empty())
    {
        return;
    }
    std::istringstream iss(serialized);
    if (!(iss >> *engine))
    {
        std::string reason = "Reason:\n\tfailed to deserialize ";
        reason += state_name;
        reason += "\n";
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand, error_by,
                                      reason.c_str());
    }
}

float Read_Output_Field_Float(const char* key, float fallback)
{
    auto iter = controller.outputs_content.find(key);
    if (iter == controller.outputs_content.end() || iter->second == "****")
    {
        return fallback;
    }
    return static_cast<float>(atof(iter->second.c_str()));
}

struct Main_Legacy_Neighbor_List_Need
{
    bool needed = false;
    bool pbc_enabled = false;
    bool selective_direct = false;
    bool clustered_direct_active = false;
    int direct_solvent_numbers = 0;
    bool lj_direct_legacy = false;
    bool lj_soft_direct_legacy = false;
    bool solvent_lj_legacy = false;
    bool pairwise_legacy = false;
    bool reaxff_legacy = false;
    bool full_legacy = false;
};

struct Main_Legacy_Neighbor_List_State
{
    bool valid = false;
    bool stale = true;
    int generation = 0;
    int stale_generation = 0;
};

Main_Legacy_Neighbor_List_State main_legacy_neighbor_list_state;

bool Main_Legacy_Neighbor_List_Trace_Enabled()
{
    const char* enabled = std::getenv("SPONGE_LEGACY_NEIGHBOR_LIST_TRACE");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

bool Main_Custom_Pairwise_Clustered_Enabled()
{
    static const bool enabled = Main_Finite_Lifecycle_Probe_Env_Enabled(
        "SPONGE_CUSTOM_PAIRWISE_CLUSTERED_NATIVE");
    return enabled;
}

const char* Main_Neighbor_List_Update_Name(int update)
{
    if (update == NEIGHBOR_LIST::FORCED_UPDATE)
    {
        return "forced";
    }
    if (update == NEIGHBOR_LIST::CONDITIONAL_UPDATE)
    {
        return "conditional";
    }
    return "none";
}

Main_Legacy_Neighbor_List_Need Main_Get_Legacy_Neighbor_List_Need()
{
    Main_Legacy_Neighbor_List_Need need;
    need.pbc_enabled = md_info.pbc.pbc != 0;
    if (!need.pbc_enabled || !neighbor_list.is_initialized)
    {
        return need;
    }

    const bool sits_clustered_direct =
        selective_interaction.Has_SITS_Direct_LJ_Coulomb() &&
        !selective_interaction.Has_REST2_Direct_LJ_Coulomb();
    need.selective_direct =
        selective_interaction.Has_Direct_LJ_Coulomb() &&
        !sits_clustered_direct;
    need.clustered_direct_active =
        sits_clustered_direct ||
        (!selective_interaction.Has_Direct_LJ_Coulomb() &&
         (lj.Use_Clustered_Direct() || lj_soft.Use_Clustered_Direct()));
    need.direct_solvent_numbers = need.clustered_direct_active
                                      ? 0
                                      : solvent_lj.local_solvent_numbers;
    need.lj_direct_legacy = !need.selective_direct && lj.is_initialized &&
                            !lj.Use_Clustered_Direct();
    need.lj_soft_direct_legacy = !need.selective_direct &&
                                 lj_soft.is_initialized &&
                                 !lj_soft.Use_Clustered_Direct();
    need.solvent_lj_legacy = need.direct_solvent_numbers > 0;
    need.pairwise_legacy =
        pairwise_force.is_initialized &&
        !Main_Custom_Pairwise_Clustered_Enabled();
    need.reaxff_legacy = reaxff.is_initialized != 0;
    need.full_legacy = neighbor_list.is_needed_full;
    need.needed = need.selective_direct || need.lj_direct_legacy ||
                  need.lj_soft_direct_legacy || need.solvent_lj_legacy ||
                  need.pairwise_legacy || need.reaxff_legacy ||
                  need.full_legacy;
    return need;
}

bool Main_Needs_Legacy_Neighbor_List()
{
    return Main_Get_Legacy_Neighbor_List_Need().needed;
}

void Main_Trace_Legacy_Neighbor_List(
    const char* site, const char* action,
    const Main_Legacy_Neighbor_List_Need& need, int requested_update,
    int effective_update)
{
    if (!Main_Legacy_Neighbor_List_Trace_Enabled())
    {
        return;
    }
    controller.MPI_printf(
        "[legacy neighbor-list gate] site=%s step=%d action=%s need=%d "
        "valid=%d stale=%d generation=%d stale_generation=%d "
        "requested_update=%s effective_update=%s pbc=%d "
        "clustered_direct_active=%d direct_solvent_numbers=%d "
        "selective_direct=%d lj_direct_legacy=%d "
        "lj_soft_direct_legacy=%d solvent_lj_legacy=%d pairwise=%d "
        "reaxff=%d full=%d legacy_deref=%s\n",
        site, md_info.sys.steps, action, need.needed ? 1 : 0,
        main_legacy_neighbor_list_state.valid ? 1 : 0,
        main_legacy_neighbor_list_state.stale ? 1 : 0,
        main_legacy_neighbor_list_state.generation,
        main_legacy_neighbor_list_state.stale_generation,
        Main_Neighbor_List_Update_Name(requested_update),
        Main_Neighbor_List_Update_Name(effective_update),
        need.pbc_enabled ? 1 : 0, need.clustered_direct_active ? 1 : 0,
        need.direct_solvent_numbers, need.selective_direct ? 1 : 0,
        need.lj_direct_legacy ? 1 : 0,
        need.lj_soft_direct_legacy ? 1 : 0,
        need.solvent_lj_legacy ? 1 : 0, need.pairwise_legacy ? 1 : 0,
        need.reaxff_legacy ? 1 : 0, need.full_legacy ? 1 : 0,
        need.needed ? "legacy-consumer" : "none");
}

void Main_Trace_Legacy_Neighbor_List_Adapter(
    const char* site, const Main_Legacy_Neighbor_List_Need& need,
    const char* outcome, const char* reason)
{
    if (!Main_Legacy_Neighbor_List_Trace_Enabled())
    {
        return;
    }
    controller.MPI_printf(
        "[legacy neighbor-list adapter] site=%s step=%d outcome=%s "
        "reason=%s clustered_direct_active=%d selective_direct=%d "
        "lj_direct_legacy=%d lj_soft_direct_legacy=%d solvent_lj_legacy=%d "
        "pairwise=%d reaxff=%d full=%d\n",
        site, md_info.sys.steps, outcome != NULL ? outcome : "unknown",
        reason != NULL ? reason : "none",
        need.clustered_direct_active ? 1 : 0,
        need.selective_direct ? 1 : 0, need.lj_direct_legacy ? 1 : 0,
        need.lj_soft_direct_legacy ? 1 : 0,
        need.solvent_lj_legacy ? 1 : 0, need.pairwise_legacy ? 1 : 0,
        need.reaxff_legacy ? 1 : 0, need.full_legacy ? 1 : 0);
}

bool Main_Try_Derived_Legacy_Neighbor_View(
    const char* site, const Main_Legacy_Neighbor_List_Need& need)
{
    if (!need.clustered_direct_active)
    {
        Main_Trace_Legacy_Neighbor_List_Adapter(
            site, need, "fallback-grid", "clustered direct is inactive");
        return false;
    }

    LJ_CLUSTERED_DIRECT_CACHE* clustered_cache = NULL;
    if (lj.Use_Clustered_Direct())
    {
        clustered_cache = lj.clustered_direct_cache;
    }
    else if (lj_soft.Use_Clustered_Direct())
    {
        clustered_cache = lj_soft.clustered_direct_cache;
    }

    LJ_CLUSTERED_LEGACY_NEIGHBOR_VIEW_REQUEST request;
    request.request_half = true;
    request.request_full = need.full_legacy || need.reaxff_legacy;
    request.contains_non_lj_consumer = need.pairwise_legacy;
    request.require_all_local_atoms = true;
    request.require_local_ghost_pairs = true;
    request.require_exclusions = true;
    request.local_atom_numbers = dd.atom_numbers;
    request.ghost_numbers = dd.ghost_numbers;
    request.cutoff = neighbor_list.cutoff;
    request.skin = neighbor_list.skin;
    request.d_atom_local = dd.atom_local;
    request.d_excluded_list_start = md_info.nb.d_excluded_list_start;
    request.d_excluded_list = md_info.nb.d_excluded_list;
    request.d_excluded_numbers = md_info.nb.d_excluded_numbers;

    const char* fallback_reason = NULL;
    const bool ready = Ensure_Legacy_Neighbor_View_From_Clustered_Payload(
        clustered_cache, request, neighbor_list.d_nl,
        neighbor_list.max_neighbor_numbers,
        neighbor_list.d_neighbor_list_overflow, &fallback_reason);
    Main_Trace_Legacy_Neighbor_List_Adapter(
        site, need, ready ? "derived-view" : "fallback-grid",
        fallback_reason);
    return ready;
}

void Main_Mark_Legacy_Neighbor_List_Stale(
    const char* site, const Main_Legacy_Neighbor_List_Need& need,
    int requested_update)
{
    if (!main_legacy_neighbor_list_state.stale)
    {
        main_legacy_neighbor_list_state.stale_generation += 1;
    }
    main_legacy_neighbor_list_state.valid = false;
    main_legacy_neighbor_list_state.stale = true;
    Main_Trace_Legacy_Neighbor_List(site, "skip-stale", need,
                                    requested_update, -1);
}

void Main_Update_Legacy_Neighbor_List_If_Needed(const char* site,
                                                int requested_update)
{
    const bool needs_legacy_neighbor_list =
        Main_Needs_Legacy_Neighbor_List();
    const Main_Legacy_Neighbor_List_Need need =
        Main_Get_Legacy_Neighbor_List_Need();
    if (!needs_legacy_neighbor_list)
    {
        Main_Mark_Legacy_Neighbor_List_Stale(site, need, requested_update);
        return;
    }

    const int effective_update = main_legacy_neighbor_list_state.valid
                                     ? requested_update
                                     : NEIGHBOR_LIST::FORCED_UPDATE;
    if (Main_Try_Derived_Legacy_Neighbor_View(site, need))
    {
        Main_Trace_Legacy_Neighbor_List(site, "derived-view", need,
                                        requested_update, effective_update);
        main_legacy_neighbor_list_state.valid = true;
        main_legacy_neighbor_list_state.stale = false;
        main_legacy_neighbor_list_state.generation += 1;
        return;
    }

    Main_Trace_Legacy_Neighbor_List(site, "update-grid", need,
                                    requested_update, effective_update);
    neighbor_list.Update(
        dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
        md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
        effective_update, md_info.nb.d_excluded_list_start,
        md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    main_legacy_neighbor_list_state.valid = true;
    main_legacy_neighbor_list_state.stale = false;
    main_legacy_neighbor_list_state.generation += 1;
}

void Main_Populate_Core_Output_Content()
{
    md_info.Step_Print(&controller);
    controller.Step_Print("potential", dd.h_sum_ene_total);
}

void Main_Probe_Current_Exchange_Observables()
{
    const int saved_write_mdout_interval = md_info.output.write_mdout_interval;
    const bool saved_print_zeroth_frame = md_info.output.print_zeroth_frame;
    const bool saved_print_virial = md_info.output.print_virial;

    md_info.output.write_mdout_interval = 1;
    md_info.output.print_zeroth_frame = true;
    md_info.output.print_virial = (md_info.mode == md_info.NPT);

    Main_Sync_Dynamic_Targets_To_Controllers();
    Main_Calculate_Force();
    dd.Get_Ek_and_Temperature(&controller, &md_info);
    dd.Get_Potential(&controller, &md_info);
    if (md_info.mode == md_info.NPT)
    {
        md_info.need_pressure = 1;
        md_info.Get_pressure(&controller, dd.atom_numbers, dd.vel, dd.d_mass,
                             dd.d_virial, main_stream);
    }

    md_info.output.write_mdout_interval = saved_write_mdout_interval;
    md_info.output.print_zeroth_frame = saved_print_zeroth_frame;
    md_info.output.print_virial = saved_print_virial;
}

}  // namespace

void Main_Run_Current_Step(bool emit_output)
{
    if (Main_Is_Finished())
    {
        return;
    }

    Main_Finite_Lifecycle_Probe("run_step_begin");
    Main_Sync_Dynamic_Targets_To_Controllers();
    Main_Finite_Lifecycle_Probe("after_sync_dynamic_targets");
    Main_Calculate_Force();
    Main_Finite_Lifecycle_Probe("after_calculate_force");
    Main_Iteration();
    Main_Finite_Lifecycle_Probe("after_iteration");
    if (emit_output)
    {
        Main_Print();
        Main_Finite_Lifecycle_Probe("after_print");
    }
    md_info.sys.steps++;
    Main_Finite_Lifecycle_Probe("after_step_increment");
}

bool Main_Is_Finished() { return md_info.sys.steps > md_info.sys.step_limit; }

void Main_Set_Step_Limit(int step_limit)
{
    if (step_limit > 0)
    {
        md_info.sys.step_limit = step_limit;
    }
}

sponge::SchedulerSnapshot Main_Get_Scheduler_Snapshot()
{
    Main_Probe_Current_Exchange_Observables();
    Main_Populate_Core_Output_Content();
    sponge::SchedulerSnapshot snapshot;
    snapshot.next_step = md_info.sys.steps;
    snapshot.last_completed_step = md_info.sys.steps - 1;
    snapshot.step_limit = md_info.sys.step_limit;
    snapshot.current_time_ps = md_info.sys.Get_Current_Time(false);
    snapshot.dt_ps = md_info.sys.dt_in_ps;
    snapshot.temperature = md_info.sys.h_temperature;
    snapshot.target_temperature = md_info.sys.target_temperature;
    snapshot.pressure = md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION;
    snapshot.target_pressure = md_info.sys.target_pressure;
    snapshot.total_potential = dd.h_sum_ene_total;
    snapshot.effective_potential = md_info.sys.h_potential;
    snapshot.box_length = {md_info.sys.box_length.x, md_info.sys.box_length.y,
                           md_info.sys.box_length.z};
    snapshot.initialized = controller.is_initialized != 0;
    snapshot.finished = Main_Is_Finished();
    return snapshot;
}

sponge::RuntimeState Main_Export_Runtime_State()
{
    sponge::RuntimeState state;
    if (!controller.is_initialized || !md_info.is_initialized)
    {
        return state;
    }

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }
    md_info.Crd_Vel_Device_To_Host(1);
    state.atom_count = md_info.atom_numbers;
    state.step = md_info.sys.steps;
    state.step_limit = md_info.sys.step_limit;
    state.start_time_ps = md_info.sys.start_time;
    state.current_time_ps = md_info.sys.Get_Current_Time(false);
    state.box_length = {md_info.sys.box_length.x, md_info.sys.box_length.y,
                        md_info.sys.box_length.z};
    state.box_angle = {md_info.sys.box_angle.x, md_info.sys.box_angle.y,
                       md_info.sys.box_angle.z};
    state.coordinates.reserve(md_info.atom_numbers);
    state.velocities.reserve(md_info.atom_numbers);
    for (int i = 0; i < md_info.atom_numbers; i++)
    {
        state.coordinates.push_back(
            Make_Runtime_State_Atom(md_info.coordinate[i]));
        state.velocities.push_back(
            Make_Runtime_State_Atom(md_info.velocity[i]));
    }
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        state.local_accelerations =
            Copy_Device_Vector_Array_To_Runtime_State(dd.acc, dd.atom_numbers);
        state.has_local_accelerations = true;
    }
    if (nhc.is_initialized)
    {
        deviceMemcpy(nhc.h_coordinate, nhc.coordinate,
                     sizeof(float) * nhc.chain_length,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(nhc.h_velocity, nhc.velocity,
                     sizeof(float) * (nhc.chain_length + 1),
                     deviceMemcpyDeviceToHost);
        state.nhc_coordinates.assign(nhc.h_coordinate,
                                     nhc.h_coordinate + nhc.chain_length);
        state.nhc_velocities.assign(nhc.h_velocity,
                                    nhc.h_velocity + nhc.chain_length + 1);
        state.has_nhc_state = true;
    }
    if (settle.is_initialized)
    {
        state.settle_last_pair_ab = Copy_Device_Vector_Array_To_Runtime_State(
            settle.last_pair_AB, settle.num_pair_local);
        state.settle_last_triangle_ba =
            Copy_Device_Vector_Array_To_Runtime_State(
                settle.last_triangle_BA, settle.num_triangle_local);
        state.settle_last_triangle_ca =
            Copy_Device_Vector_Array_To_Runtime_State(
                settle.last_triangle_CA, settle.num_triangle_local);
        state.has_settle_state = true;
    }
    if (shake.is_initialized)
    {
        state.shake_last_pair_dr = Copy_Device_Vector_Array_To_Runtime_State(
            shake.last_pair_dr, constrain.num_pair_local);
        state.has_shake_state = true;
    }
    if (press_baro.is_initialized)
    {
        state.pressure_barostat_g = {press_baro.g.a11, press_baro.g.a21,
                                     press_baro.g.a22, press_baro.g.a31,
                                     press_baro.g.a32, press_baro.g.a33};
        state.pressure_barostat_v0 = press_baro.V0;
        state.pressure_barostat_rng_state =
            Serialize_Stream_State(press_baro.generator);
        state.pressure_barostat_distribution_state =
            Serialize_Stream_State(press_baro.distribution);
        state.has_pressure_barostat_state = true;
    }
    if (mc_baro.is_initialized)
    {
        state.mc_barostat_total_count = {mc_baro.total_count[0],
                                         mc_baro.total_count[1],
                                         mc_baro.total_count[2]};
        state.mc_barostat_accept_count = {mc_baro.accep_count[0],
                                          mc_baro.accep_count[1],
                                          mc_baro.accep_count[2]};
        state.mc_barostat_accept_rate = {mc_baro.accept_rate[0],
                                         mc_baro.accept_rate[1],
                                         mc_baro.accept_rate[2]};
        state.mc_barostat_delta_box_length_max = {
            mc_baro.Delta_Box_Length_Max[0], mc_baro.Delta_Box_Length_Max[1],
            mc_baro.Delta_Box_Length_Max[2]};
        state.mc_barostat_rng_state = Serialize_Stream_State(mc_baro.generator);
        state.has_mc_barostat_state = true;
    }
    if (middle_langevin.is_initialized)
    {
        state.middle_langevin_rng_state = Copy_Device_Bytes_To_Host(
            middle_langevin.rand_state, middle_langevin.float4_numbers);
        state.has_middle_langevin_rng_state = true;
    }
    if (ad_thermo.is_initialized)
    {
        state.andersen_rng_state = Copy_Device_Bytes_To_Host(
            ad_thermo.rand_state, ad_thermo.float4_numbers);
        state.has_andersen_rng_state = true;
    }
    if (bussi_thermo.is_initialized)
    {
        state.bussi_rng_state = Serialize_Stream_State(bussi_thermo.e);
        state.bussi_distribution_state =
            Serialize_Stream_State(bussi_thermo.normal01);
        state.has_bussi_rng_state = true;
    }
    state.valid = true;
    return state;
}

void Main_Import_Runtime_State(const sponge::RuntimeState& state)
{
    if (!controller.is_initialized || !md_info.is_initialized)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Import_Runtime_State",
            "Reason:\n\tSPONGE runtime is not initialized\n");
    }
    if (!state.valid)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Import_Runtime_State",
            "Reason:\n\tinput runtime state is invalid\n");
    }
    if (state.atom_count != md_info.atom_numbers)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Import_Runtime_State",
            "Reason:\n\tatom count mismatch while importing runtime state\n");
    }
    if (static_cast<int>(state.coordinates.size()) != md_info.atom_numbers ||
        static_cast<int>(state.velocities.size()) != md_info.atom_numbers)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Import_Runtime_State",
            "Reason:\n\tcoordinate/velocity array size mismatch while "
            "importing runtime state\n");
    }

    for (int i = 0; i < md_info.atom_numbers; i++)
    {
        md_info.coordinate[i] = Make_Vector(state.coordinates[i]);
        md_info.velocity[i] = Make_Vector(state.velocities[i]);
    }
    deviceMemcpy(md_info.crd, md_info.coordinate,
                 sizeof(VECTOR) * md_info.atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(md_info.vel, md_info.velocity,
                 sizeof(VECTOR) * md_info.atom_numbers,
                 deviceMemcpyHostToDevice);

    md_info.sys.steps = state.step;
    md_info.sys.step_limit = state.step_limit;
    md_info.sys.start_time = state.start_time_ps;
    md_info.sys.current_time = state.current_time_ps;
    md_info.sys.box_length = {state.box_length[0], state.box_length[1],
                              state.box_length[2]};
    md_info.sys.box_angle = {state.box_angle[0], state.box_angle[1],
                             state.box_angle[2]};
    if (state.has_nhc_state && nhc.is_initialized)
    {
        if (static_cast<int>(state.nhc_coordinates.size()) !=
                nhc.chain_length ||
            static_cast<int>(state.nhc_velocities.size()) !=
                nhc.chain_length + 1)
        {
            controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Main_Import_Runtime_State",
                "Reason:\n\tNHC state size mismatch while importing runtime "
                "state\n");
        }
        for (int i = 0; i < nhc.chain_length; i++)
        {
            nhc.h_coordinate[i] = state.nhc_coordinates[i];
        }
        for (int i = 0; i < nhc.chain_length + 1; i++)
        {
            nhc.h_velocity[i] = state.nhc_velocities[i];
        }
        deviceMemcpy(nhc.coordinate, nhc.h_coordinate,
                     sizeof(float) * nhc.chain_length,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(nhc.velocity, nhc.h_velocity,
                     sizeof(float) * (nhc.chain_length + 1),
                     deviceMemcpyHostToDevice);
    }
    if (state.has_pressure_barostat_state && press_baro.is_initialized)
    {
        press_baro.g = {
            state.pressure_barostat_g[0], state.pressure_barostat_g[1],
            state.pressure_barostat_g[2], state.pressure_barostat_g[3],
            state.pressure_barostat_g[4], state.pressure_barostat_g[5]};
        press_baro.V0 = state.pressure_barostat_v0;
        Deserialize_Stream_State(
            state.pressure_barostat_rng_state, &press_baro.generator,
            "Main_Import_Runtime_State", "pressure barostat generator state");
        Deserialize_Stream_State(state.pressure_barostat_distribution_state,
                                 &press_baro.distribution,
                                 "Main_Import_Runtime_State",
                                 "pressure barostat distribution state");
    }
    if (state.has_mc_barostat_state && mc_baro.is_initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            mc_baro.total_count[i] = state.mc_barostat_total_count[i];
            mc_baro.accep_count[i] = state.mc_barostat_accept_count[i];
            mc_baro.accept_rate[i] = state.mc_barostat_accept_rate[i];
            mc_baro.Delta_Box_Length_Max[i] =
                state.mc_barostat_delta_box_length_max[i];
        }
        Deserialize_Stream_State(
            state.mc_barostat_rng_state, &mc_baro.generator,
            "Main_Import_Runtime_State", "MC barostat RNG state");
    }
    if (state.has_middle_langevin_rng_state && middle_langevin.is_initialized)
    {
        Copy_Host_Bytes_To_Device(
            state.middle_langevin_rng_state, middle_langevin.rand_state,
            middle_langevin.float4_numbers, "Main_Import_Runtime_State",
            "middle_langevin RNG state");
    }
    if (state.has_andersen_rng_state && ad_thermo.is_initialized)
    {
        Copy_Host_Bytes_To_Device(
            state.andersen_rng_state, ad_thermo.rand_state,
            ad_thermo.float4_numbers, "Main_Import_Runtime_State",
            "Andersen RNG state");
    }
    if (state.has_bussi_rng_state && bussi_thermo.is_initialized)
    {
        Deserialize_Stream_State(state.bussi_rng_state, &bussi_thermo.e,
                                 "Main_Import_Runtime_State",
                                 "Bussi RNG state");
        Deserialize_Stream_State(
            state.bussi_distribution_state, &bussi_thermo.normal01,
            "Main_Import_Runtime_State", "Bussi distribution state");
    }
    md_info.pbc.PBC_Check();
    md_info.output.current_crd_synchronized_step = -1;
    Main_Refresh_Local_State(true);
    if (state.has_local_accelerations &&
        CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Copy_Runtime_State_Vector_Array_To_Device(
            state.local_accelerations, dd.acc, dd.atom_numbers,
            "Main_Import_Runtime_State", "local accelerations");
    }
    if (state.has_settle_state && settle.is_initialized)
    {
        Copy_Runtime_State_Vector_Array_To_Device(
            state.settle_last_pair_ab, settle.last_pair_AB,
            settle.num_pair_local, "Main_Import_Runtime_State",
            "SETTLE last pair AB");
        Copy_Runtime_State_Vector_Array_To_Device(
            state.settle_last_triangle_ba, settle.last_triangle_BA,
            settle.num_triangle_local, "Main_Import_Runtime_State",
            "SETTLE last triangle BA");
        Copy_Runtime_State_Vector_Array_To_Device(
            state.settle_last_triangle_ca, settle.last_triangle_CA,
            settle.num_triangle_local, "Main_Import_Runtime_State",
            "SETTLE last triangle CA");
    }
    if (state.has_shake_state && shake.is_initialized)
    {
        Copy_Runtime_State_Vector_Array_To_Device(
            state.shake_last_pair_dr, shake.last_pair_dr,
            constrain.num_pair_local, "Main_Import_Runtime_State",
            "SHAKE last pair dr");
    }
}

sponge::WorkerExchangeObservable Main_Collect_Exchange_Observables()
{
    Main_Probe_Current_Exchange_Observables();
    sponge::WorkerExchangeObservable observable;
    observable.step = md_info.sys.steps;
    observable.time_ps = md_info.sys.Get_Current_Time(false);
    observable.total_potential = dd.h_sum_ene_total;
    observable.effective_potential = md_info.sys.h_potential;
    observable.temperature = md_info.sys.h_temperature;
    observable.target_temperature = md_info.sys.target_temperature;
    observable.pressure = md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION;
    observable.target_pressure = md_info.sys.target_pressure;
    observable.volume = md_info.sys.Get_Volume();
    return observable;
}

void Main_Ensure_Foreign_State_Probe_Safe()
{
    if (meta.is_initialized)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Main_Ensure_Foreign_State_Probe_Safe",
            "Reason:\n\tforeign-state observable probing is not safe when "
            "sink metadynamics bias is enabled, because the probe worker does "
            "not serialize/import metadynamics history state yet.\n");
    }
    if (!selective_interaction.Is_Probe_Safe())
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Main_Ensure_Foreign_State_Probe_Safe",
            "Reason:\n\tforeign-state observable probing is not safe when "
            "SITS/enhanced-sampling bias is enabled, because the probe worker "
            "does not serialize/import SITS bias history state yet.\n");
    }
}

void Main_Scale_Velocities(float factor)
{
    if (factor == 1.0f)
    {
        return;
    }
    md_info.Crd_Vel_Device_To_Host(1);
    for (int i = 0; i < md_info.atom_numbers; i++)
    {
        md_info.velocity[i] = factor * md_info.velocity[i];
    }
    deviceMemcpy(md_info.vel, md_info.velocity,
                 sizeof(VECTOR) * md_info.atom_numbers,
                 deviceMemcpyHostToDevice);
    md_info.output.current_crd_synchronized_step = -1;
}

void Main_Invalidate_Neighbor_List(bool rebuild_dd)
{
    Main_Refresh_Local_State(rebuild_dd);
}

namespace sponge
{

SpongeScheduler::~SpongeScheduler()
{
    if (initialized_ && !finalized_)
    {
        Finalize();
    }
}

void SpongeScheduler::InitializeFromArgv(int argc, char** argv)
{
    owned_args_.clear();
    owned_args_.reserve(argc > 0 ? argc : 1);
    if (argc <= 0)
    {
        owned_args_.push_back("SPONGE");
    }
    else
    {
        for (int i = 0; i < argc; i++)
        {
            owned_args_.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        }
    }
    InitializeFromOwnedArgs();
}

void SpongeScheduler::InitializeFromArgs(const std::vector<std::string>& args)
{
    owned_args_.clear();
    owned_args_.reserve(args.size() + 1);
    if (args.empty() || (!args.front().empty() && args.front()[0] == '-'))
    {
        owned_args_.push_back("SPONGE");
    }
    owned_args_.insert(owned_args_.end(), args.begin(), args.end());
    InitializeFromOwnedArgs();
}

void SpongeScheduler::InitializeFromOwnedArgs()
{
    if (initialized_ && !finalized_)
    {
        throw std::runtime_error(
            "SpongeScheduler::Initialize called on an active runtime");
    }

    RebuildArgvCache();
    finalized_ = false;
    Main_Initial(static_cast<int>(argv_cache_.size()), argv_cache_.data());
    initialized_ = true;
}

void SpongeScheduler::RunSingleStep(bool emit_output)
{
    EnsureInitialized("RunSingleStep");
    Main_Run_Current_Step(emit_output);
}

void SpongeScheduler::RunSteps(int steps, bool emit_output)
{
    EnsureInitialized("RunSteps");
    if (steps <= 0)
    {
        return;
    }
    for (int i = 0; i < steps && !Main_Is_Finished(); i++)
    {
        Main_Run_Current_Step(emit_output);
    }
}

void SpongeScheduler::RunToEnd(bool emit_output)
{
    EnsureInitialized("RunToEnd");
    while (!Main_Is_Finished())
    {
        Main_Run_Current_Step(emit_output);
    }
}

void SpongeScheduler::SetStepLimit(int step_limit)
{
    EnsureInitialized("SetStepLimit");
    Main_Set_Step_Limit(step_limit);
}

SchedulerSnapshot SpongeScheduler::Snapshot() const
{
    EnsureInitialized("Snapshot");
    return Main_Get_Scheduler_Snapshot();
}

RuntimeState SpongeScheduler::ExportRuntimeState()
{
    EnsureInitialized("ExportRuntimeState");
    return Main_Export_Runtime_State();
}

void SpongeScheduler::ImportRuntimeState(const RuntimeState& state)
{
    EnsureInitialized("ImportRuntimeState");
    Main_Import_Runtime_State(state);
}

WorkerExchangeObservable SpongeScheduler::CollectExchangeObservables() const
{
    EnsureInitialized("CollectExchangeObservables");
    return Main_Collect_Exchange_Observables();
}

void SpongeScheduler::EnsureForeignStateProbeSafe() const
{
    EnsureInitialized("EnsureForeignStateProbeSafe");
    Main_Ensure_Foreign_State_Probe_Safe();
}

void SpongeScheduler::ScaleVelocities(float factor)
{
    EnsureInitialized("ScaleVelocities");
    Main_Scale_Velocities(factor);
}

void SpongeScheduler::InvalidateNeighborList(bool rebuild_dd)
{
    EnsureInitialized("InvalidateNeighborList");
    Main_Invalidate_Neighbor_List(rebuild_dd);
}

bool SpongeScheduler::IsInitialized() const
{
    return initialized_ && !finalized_;
}

bool SpongeScheduler::IsFinished() const
{
    EnsureInitialized("IsFinished");
    return Main_Is_Finished();
}

void SpongeScheduler::Finalize()
{
    EnsureInitialized("Finalize");
    Main_Finite_Async_Lifecycle_Probe_Report();
    Main_Clear();
    initialized_ = false;
    finalized_ = true;
}

void SpongeScheduler::EnsureInitialized(const char* caller) const
{
    if (!initialized_ || finalized_)
    {
        throw std::runtime_error(std::string("SpongeScheduler::") + caller +
                                 " called before initialization");
    }
}

void SpongeScheduler::RebuildArgvCache()
{
    argv_cache_.clear();
    argv_cache_.reserve(owned_args_.size());
    for (std::string& arg : owned_args_)
    {
        argv_cache_.push_back(arg.data());
    }
}

}  // namespace sponge

void Main_Initial(int argc, char* argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    Xponge::system.Load_Inputs(&controller);
    cv_controller.Initial(&controller,
                          &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    controller.Step_Print_Initial("potential", "%.2f");
    controller.Step_Print_Initial("eff_pot", "%.7e");
    qc.Initial(&controller, md_info.atom_numbers, md_info.crd);
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);

    if (md_info.mode >= md_info.NVT &&
        (!controller.Command_Exist("thermostat") &&
         !controller.Command_Exist("thermostat_mode")))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if (THERMOSTAT_IS("middle_langevin") || THERMOSTAT_IS("langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers,
                                md_info.sys.target_temperature, md_info.h_mass);
    }
    else if (THERMOSTAT_IS("andersen"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature,
                          md_info.atom_numbers, md_info.sys.dt_in_ps,
                          md_info.h_mass);
    }
    else if (THERMOSTAT_IS("bussi_thermostat"))
    {
        bussi_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (THERMOSTAT_IS("berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (THERMOSTAT_IS("nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.atom_numbers,
                    md_info.sys.target_temperature, md_info.h_mass);
    }

    if (md_info.mode == md_info.NPT && !controller.Command_Exist("barostat") &&
        !controller.Command_Exist("barostat_mode"))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tbarostat is required for NPT simulations\n");
    }
    if (BAROSTAT_IS("andersen_barostat") || BAROSTAT_IS("bussi_barostat") ||
        BAROSTAT_IS("berendsen_barostat"))
    {
        press_baro.Initial(&controller, md_info.sys.target_pressure,
                           md_info.pbc.cell, &Main_Box_Change);
    }
    if (BAROSTAT_IS("monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers,
                        md_info.sys.target_pressure, md_info.sys.box_length,
                        md_info.pbc.cell);
    }

    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff);
        lj_soft.Initial(&controller, md_info.nb.cutoff);
        pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
                   md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
                   md_info.no_direct_interaction_virtual_atom_numbers);
        pairwise_force.Initial(&controller);
        if (pairwise_force.is_initialized &&
            Main_Custom_Pairwise_Clustered_Enabled())
        {
            pairwise_clustered_cache =
                Acquire_Shared_LJ_Clustered_Direct_Cache(
                    &controller, "clustered_spatial_service", false, true);
        }
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);

        selective_interaction.Initial(&controller, md_info.atom_numbers);
        if (selective_interaction.Uses_SITS_Listed_Forces())
        {
            sits_dihedral.Initial(&controller, "sits_dihedral");
            sits_nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B,
                              lj.h_atom_LJ_type, "sits_nb14");
            sits_cmap.Initial(&controller, "sits_cmap");
        }
        selective_interaction.Check_Solvent(&controller, md_info.atom_numbers,
                                            solvent_lj.solvent_numbers);
        if (selective_interaction.Has_SITS_Direct_LJ_Coulomb() &&
            !selective_interaction.Has_REST2_Direct_LJ_Coulomb())
        {
            if (lj_soft.is_initialized)
            {
                lj_soft.Enable_Clustered_Direct();
            }
            else
            {
                lj.Enable_Clustered_Direct();
            }
        }
    }
    else
    {
        LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
        CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
        if (controller.Command_Exist("gb", "in_file"))
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B,
                     LJ_NOPBC.h_atom_LJ_type);
        selective_interaction.Initial(&controller, md_info.atom_numbers);
    }

    bond.Initial(&controller, &md_info.sys.connectivity,
                 &md_info.sys.connected_distance);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    cmap.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    listed_forces.Initial(&controller, &md_info.sys.connectivity,
                          &md_info.sys.connected_distance);

    sw.Initial(&controller, "SW", &neighbor_list.is_needed_full);
    edip.Initial(&controller, "EDIP", &neighbor_list.is_needed_full);
    eam.Initial(&controller, md_info.atom_numbers, "EAM",
                &neighbor_list.is_needed_full);
    tersoff.Initial(&controller, md_info.atom_numbers, "TERSOFF",
                    &neighbor_list.is_needed_full);
    reaxff.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                   &neighbor_list.cutoff_full, &neighbor_list.is_needed_full);

    restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    hard_wall.Initial(&controller, md_info.sys.target_temperature,
                      md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    if (controller.Command_Exist("constrain_mode"))
    {
        constrain.Initial_List(&controller, md_info.sys.connected_distance,
                               md_info.h_mass);
        constrain.Initial_Constrain(&controller, md_info.atom_numbers,
                                    md_info.dt, md_info.sys.box_length,
                                    md_info.h_mass, &md_info.sys.freedom);
        settle.Initial(&controller, &constrain, md_info.h_mass);
        if (controller.Command_Choice("constrain_mode", "SHAKE"))
        {
            shake.Initial_SHAKE(&controller, &constrain);
        }
        if (md_info.mode == md_info.MINIMIZATION)
        {
            constrain.v_factor = 0.0f;
        }
        if (middle_langevin.is_initialized)
        {
            constrain.v_factor = middle_langevin.exp_gamma;
            constrain.x_factor = 0.5 * (1. + middle_langevin.exp_gamma);
        }
    }
    vatom.Initial(&controller, &cv_controller, md_info.atom_numbers,
                  md_info.no_direct_interaction_virtual_atom_numbers,
                  cv_controller.cv_vatom_name, md_info.h_mass,
                  &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.crd, md_info.pbc.cell, md_info.pbc.rcell);

    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers,
                              md_info.nb.cutoff, md_info.nb.skin,
                              md_info.pbc.cell, md_info.pbc.rcell);
    }
    steer_cv.Initial(&controller, &cv_controller);
    restrain_cv.Initial(&controller, &cv_controller);
    meta.Initial(&controller, &cv_controller);

    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();

    md_info.ug.Initial_Edge(md_info.atom_numbers);
    constrain.update_ug_connectivity(&md_info.ug.connectivity);
    settle.update_ug_connectivity(&md_info.ug.connectivity);
    vatom.update_ug_connectivity(&md_info.ug.connectivity);
    md_info.ug.Read_Update_Group(md_info.atom_numbers);
    md_info.mol.Initial(&controller);
    if (md_info.pbc.pbc)
    {
        solvent_lj.Initial(&controller, &lj, &lj_soft, &md_info,
                           md_info.mode >= md_info.NVT);
    }
    Main_Process_Management();

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Main_Refresh_Local_State(true);
        plugin.Set_Domain_Information(&dd);
    }

    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);

    controller.Print_First_Line_To_Mdout();
}

void Main_Calculate_Force()
{
    Main_Finite_Lifecycle_Probe("force_entry");
    bool use_reaxff_eeq = reaxff.eeq.is_initialized;
    const int cv_atom_numbers =
        md_info.atom_numbers +
        md_info.no_direct_interaction_virtual_atom_numbers;
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    Main_Finite_Lifecycle_Probe("after_md_reset_force");
    qc.Solve_SCF(dd.crd, md_info.sys.box_length, true, md_info.sys.steps);
    Main_Finite_Lifecycle_Probe("after_qc_scf");
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps,
                                        &md_info.need_potential);
    press_baro.Ask_For_Calculate_Pressure(md_info.sys.steps,
                                          &md_info.need_pressure);
    if (press_baro.is_initialized && md_info.output.Check_Mdout_Step())
    {
        md_info.need_pressure = 1;
    }
    if (bd_thermo.is_initialized || bussi_thermo.is_initialized ||
        nhc.is_initialized)
    {
        md_info.need_kinetic = 1;
    }
    selective_interaction.Reset_Force_Energy(&md_info.need_potential);

    controller.Get_Time_Recorder("Calculate_Force")->Start();
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, false, false, true, false);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        dd.Reset_Force_and_Virial(&md_info);
        Main_Finite_Lifecycle_Probe("after_dd_reset_force");
        // QC 梯度必须在 dd.Reset_Force_and_Virial 之后调用
        if (qc.is_initialized && qc.need_gradient)
            qc.Compute_Gradient(dd.frc, dd.crd, md_info.sys.box_length,
                                md_info.need_pressure, dd.d_virial);
        Main_Finite_Lifecycle_Probe("after_qc_gradient");
        dd.Update_Ghost(&controller);
        Main_Finite_Lifecycle_Probe("after_update_ghost");
        Main_Update_Legacy_Neighbor_List_If_Needed(
            "Main_Calculate_Force", neighbor_list.CONDITIONAL_UPDATE);
        Main_Finite_Lifecycle_Probe("after_force_neighbor_update");

        reaxff.Calculate_Force(&dd, &md_info, &neighbor_list);
        Main_Finite_Lifecycle_Probe("after_reaxff_force");

        LJ_NOPBC.LJ_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.frc, md_info.need_potential,
            dd.d_energy, dd.d_excluded_list_start, dd.d_excluded_list,
            dd.d_excluded_numbers);
        CF_NOPBC.Coulomb_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.d_charge, dd.frc,
            md_info.need_potential, dd.d_energy, dd.d_excluded_list_start,
            dd.d_excluded_list, dd.d_excluded_numbers);
        gb.Get_Effective_Born_Radius(dd.crd);
        gb.GB_Force_With_Atom_Energy(dd.atom_numbers, dd.crd, dd.d_charge,
                                     dd.frc, dd.d_energy);

        if (!use_reaxff_eeq)
        {
            pm.MPI_PME_Excluded_Force_With_Atom_Energy(
                dd.atom_numbers, dd.atom_local, dd.atom_local_id, dd.crd,
                md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                dd.d_excluded_list_start, dd.d_excluded_list,
                dd.d_excluded_numbers, dd.frc, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial);
        }

        if (selective_interaction.Uses_SITS_Listed_Forces())
        {
            sits_dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                selective_interaction.Select_Force(), md_info.need_potential,
                selective_interaction.Select_Atom_Energy(),
                md_info.need_pressure,
                selective_interaction.Select_Atom_Virial_Tensor());
            sits_nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
                dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell,
                selective_interaction.Select_Force(), md_info.need_potential,
                selective_interaction.Select_Atom_Energy(),
                md_info.need_pressure,
                selective_interaction.Select_Atom_Virial_Tensor());
            sits_cmap.CMAP_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                selective_interaction.Select_Force(), md_info.need_potential,
                selective_interaction.Select_Atom_Energy(),
                md_info.need_pressure,
                selective_interaction.Select_Atom_Virial_Tensor());
        }
        const bool sits_clustered_direct =
            selective_interaction.Has_SITS_Direct_LJ_Coulomb() &&
            !selective_interaction.Has_REST2_Direct_LJ_Coulomb();
        const bool clustered_direct_active =
            sits_clustered_direct ||
            (!selective_interaction.Has_Direct_LJ_Coulomb() &&
             (lj.Use_Clustered_Direct() || lj_soft.Use_Clustered_Direct()));
        const int direct_solvent_numbers =
            clustered_direct_active ? 0 : solvent_lj.local_solvent_numbers;
        if (selective_interaction.Has_Direct_LJ_Coulomb())
        {
            if (sits_clustered_direct)
            {
                const char* failure_reason = NULL;
                if (lj_soft.is_initialized)
                {
                    lj_soft
                        .LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
                            md_info.atom_numbers, dd.atom_numbers, 0,
                            dd.ghost_numbers, dd.crd, dd.d_charge, dd.frc,
                            md_info.pbc.cell, md_info.pbc.rcell,
                            neighbor_list.d_nl, pm.beta,
                            md_info.need_potential, dd.d_energy,
                            md_info.need_pressure, dd.d_virial,
                            pm.d_direct_atom_energy);
                    if (!selective_interaction
                             .LJ_Soft_Core_Direct_CF_Force_Clustered(
                                 md_info.atom_numbers, dd.atom_numbers,
                                 dd.ghost_numbers, &lj_soft, dd.frc,
                                 md_info.pbc.cell, md_info.pbc.rcell,
                                 md_info.nb.cutoff, pm.beta,
                                 md_info.need_potential, dd.d_energy,
                                 md_info.need_pressure, dd.d_virial,
                                 pm.d_direct_atom_energy,
                                 &failure_reason))
                    {
                        throw std::runtime_error(
                            std::string(
                                "clustered-native SITS soft-LJ dispatch "
                                "rejected the clustered payload: ") +
                            (failure_reason == NULL
                                 ? "unknown SITS soft-LJ failure"
                                 : failure_reason));
                    }
                }
                else
                {
                    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                        md_info.atom_numbers, dd.atom_numbers, 0,
                        dd.ghost_numbers, dd.crd, dd.d_charge, dd.frc,
                        md_info.pbc.cell, md_info.pbc.rcell,
                        neighbor_list.d_nl, pm.beta,
                        md_info.need_potential, dd.d_energy,
                        md_info.need_pressure, dd.d_virial,
                        pm.d_direct_atom_energy);
                    if (!selective_interaction.LJ_Direct_CF_Force_Clustered(
                            md_info.atom_numbers, dd.atom_numbers,
                            dd.ghost_numbers, dd.crd, dd.d_charge, &lj, dd.frc,
                            md_info.pbc.cell, md_info.pbc.rcell,
                            md_info.nb.cutoff, pm.beta, md_info.need_potential,
                            dd.d_energy, md_info.need_pressure, dd.d_virial,
                            pm.d_direct_atom_energy, &failure_reason))
                    {
                        throw std::runtime_error(
                            std::string(
                                "clustered-native SITS dispatch rejected "
                                "the clustered payload: ") +
                            (failure_reason == NULL
                                 ? "unknown SITS failure"
                                 : failure_reason));
                    }
                }
            }
            else
            {
                selective_interaction
                    .LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
                        md_info.atom_numbers, dd.atom_numbers,
                        direct_solvent_numbers, dd.ghost_numbers, dd.crd,
                        dd.d_charge, &lj, dd.frc, md_info.pbc.cell,
                        md_info.pbc.rcell, neighbor_list.d_nl,
                        md_info.nb.cutoff, pm.beta, md_info.need_potential,
                        dd.d_energy, md_info.need_pressure, dd.d_virial,
                        pm.d_direct_atom_energy);
                selective_interaction
                    .LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
                        md_info.atom_numbers, dd.atom_numbers,
                        direct_solvent_numbers, dd.ghost_numbers, dd.crd,
                        dd.d_charge, &lj_soft, dd.frc, md_info.pbc.cell,
                        md_info.pbc.rcell, neighbor_list.d_nl,
                        pm.beta, md_info.need_potential, dd.d_energy,
                        md_info.need_pressure, dd.d_virial,
                        pm.d_direct_atom_energy);
            }
        }
        else
        {
            lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                direct_solvent_numbers, dd.ghost_numbers, dd.crd, dd.d_charge,
                dd.frc, md_info.pbc.cell, md_info.pbc.rcell, neighbor_list.d_nl,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);

            lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                direct_solvent_numbers, dd.ghost_numbers, dd.crd, dd.d_charge,
                dd.frc, md_info.pbc.cell, md_info.pbc.rcell, neighbor_list.d_nl,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);
        }
        if (direct_solvent_numbers > 0)
        {
            solvent_lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                dd.atom_numbers, dd.res_numbers, dd.res_start, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, pm.beta, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);
        }

        lj.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);

        lj_soft.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);
        sw.SW_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        edip.EDIP_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        eam.EAM_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        tersoff.TERSOFF_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        listed_forces.Compute_Force(dd.atom_numbers, dd.crd, md_info.pbc.cell,
                                    md_info.pbc.rcell, dd.frc,
                                    md_info.need_potential, dd.d_energy,
                                    md_info.need_pressure, dd.d_virial);
        if (pairwise_force.is_initialized &&
            Main_Custom_Pairwise_Clustered_Enabled())
        {
            LJ_CLUSTERED_DIRECT_CACHE* clustered_cache =
                pairwise_clustered_cache;
            if (clustered_cache == NULL && lj.Use_Clustered_Direct())
            {
                clustered_cache = lj.clustered_direct_cache;
            }
            if (clustered_cache == NULL && lj_soft.Use_Clustered_Direct())
            {
                clustered_cache = lj_soft.clustered_direct_cache;
            }
            if (clustered_cache != NULL)
            {
#ifdef USE_CPU
                constexpr bool need_gmxpacked_payload = false;
                constexpr bool runtime_gmxpacked_direct = false;
#else
                constexpr bool need_gmxpacked_payload = true;
                constexpr bool runtime_gmxpacked_direct = true;
#endif
                clustered_cache->Build(
                    dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                    md_info.nb.cutoff, md_info.need_pressure != 0, false,
                    need_gmxpacked_payload, false,
                    runtime_gmxpacked_direct);
            }
            CLUSTERED_SPATIAL_VIEW clustered_view;
            const char* clustered_failure_reason = NULL;
            if (!Make_Clustered_Spatial_View_From_LJ_Cache(
                    clustered_cache, &clustered_view,
                    &clustered_failure_reason))
            {
                throw std::runtime_error(
                    std::string("clustered-native custom pairwise dispatch "
                                "requires a current clustered payload: ") +
                    (clustered_failure_reason == NULL
                         ? "unknown clustered-view failure"
                         : clustered_failure_reason));
            }
            if (!pairwise_force.Compute_Force_Clustered(
                    clustered_view, dd.crd, md_info.pbc.cell,
                    md_info.pbc.rcell, md_info.nb.cutoff, pm.beta,
                    dd.d_charge, dd.frc, md_info.need_potential, dd.d_energy,
                    md_info.need_pressure, dd.d_virial,
                    pm.d_direct_atom_energy, &clustered_failure_reason))
            {
                throw std::runtime_error(
                    std::string("clustered-native custom pairwise dispatch "
                                "rejected the clustered payload: ") +
                    (clustered_failure_reason == NULL
                         ? "unknown custom-pairwise failure"
                         : clustered_failure_reason));
            }
        }
        else
        {
            pairwise_force.Compute_Force(
                neighbor_list.d_nl, dd.crd, md_info.pbc.cell,
                md_info.pbc.rcell, md_info.nb.cutoff, pm.beta, dd.d_charge,
                dd.frc, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);
        }
        angle.Angle_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        bond.Bond_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        cmap.CMAP_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        improper.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
            dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        soft_walls.Compute_Force(dd.atom_numbers, dd.crd, dd.frc,
                                 md_info.need_potential, dd.d_energy);
        plugin.Calculate_Force();

        restrain.Restraint(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                           md_info.need_potential, dd.d_energy,
                           md_info.need_pressure, dd.d_virial, dd.frc, &md_info,
                           &dd);

        if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
        {
            vatom.Coordinate_Refresh_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            if (!use_reaxff_eeq)
            {
                pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                    dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                    dd.frc, md_info.need_pressure, md_info.need_potential,
                    dd.d_virial, dd.d_energy, md_info.sys.steps);
            }

            cv_controller.Compute_CV_For_Print(
                cv_atom_numbers, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);

            steer_cv.Steer(cv_atom_numbers, dd.crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps, dd.d_energy,
                           dd.d_virial, dd.frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                cv_atom_numbers, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, dd.d_energy, dd.d_virial, dd.frc,
                md_info.need_potential, md_info.need_pressure);
            meta.Do_Metadynamics(cv_atom_numbers, dd.crd, md_info.pbc.cell,
                                 md_info.pbc.rcell, md_info.sys.steps,
                                 md_info.need_potential, md_info.need_pressure,
                                 dd.frc, dd.d_energy, dd.d_virial,
                                 md_info.sys.h_temperature);
            vatom.Force_Redistribute_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, dd.frc);
        }
        else
        {
            if (!use_reaxff_eeq)
            {
                pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                                   dd.atom_numbers);
            }
        }
        selective_interaction.Update_And_Enhance(
            md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure,
            dd.d_virial, dd.frc,
            1.0f / (CONSTANT_kB * md_info.sys.target_temperature));
        Main_Finite_Lifecycle_Probe("after_selective_update");
        vatom.Force_Redistribute(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                                 dd.frc);
        Main_Finite_Lifecycle_Probe("after_vatom_force_redistribute");
    }
    else
    {
        if (!use_reaxff_eeq)
        {
            pm.reset_global_force(
                md_info.no_direct_interaction_virtual_atom_numbers);
            vatom.Coordinate_Refresh_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                md_info.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.d_charge, md_info.frc, md_info.need_pressure,
                md_info.need_potential, md_info.d_atom_virial_tensor,
                md_info.d_atom_energy, md_info.sys.steps);
            cv_controller.Compute_CV_For_Print(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);
            steer_cv.Steer(cv_atom_numbers, pm.g_crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps,
                           md_info.d_atom_energy, md_info.d_atom_virial_tensor,
                           pm.g_frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, pm.g_frc, md_info.need_potential,
                md_info.need_pressure);
            meta.Do_Metadynamics(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.need_potential,
                md_info.need_pressure, pm.g_frc, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, md_info.sys.h_temperature);
            vatom.Force_Redistribute_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, pm.g_frc);
            pm.add_force_g_to_l(md_info.frc);
            pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                               dd.atom_numbers);
        }
    }
    Main_Finite_Lifecycle_Probe("before_scale_force_dynamic_dt");
    md_info.min.Scale_Force_For_Dynamic_Dt(dd.atom_numbers, dd.d_mass_inverse,
                                           dd.frc, dd.vel, dd.acc);
    Main_Finite_Lifecycle_Probe("force_exit");
    controller.Get_Time_Recorder("Calculate_Force")->Stop();
}

void Main_Refresh_Local_State(bool rebuild_dd)
{
    Main_Finite_Lifecycle_Probe("refresh_entry");
    if (rebuild_dd)
    {
        dd.Send_Recv_Dom_Dec(&controller);
        dd.Find_Neighbor_Domain(&controller, &md_info);
        dd.Get_Atoms(&controller, &md_info);
        Main_Finite_Lifecycle_Probe("refresh_after_get_atoms");
    }
    dd.Get_Ghost(&controller, &md_info);
    Main_Finite_Lifecycle_Probe("refresh_after_get_ghost");
    lj.Maybe_Apply_Ordered_Layout(&controller, &dd, md_info.pbc.cell,
                                  md_info.pbc.rcell, md_info.sys.box_length);
    Main_Finite_Lifecycle_Probe("refresh_after_ordered_layout");
    dd.Get_Excluded(&controller, &md_info);
    Main_Finite_Lifecycle_Probe("refresh_after_get_excluded");

    middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
    ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
    nhc.Get_Local(dd.atom_local, dd.atom_numbers);

    lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    lj_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    solvent_lj.Get_Local(dd.res_numbers, dd.res_len, dd.atom_numbers,
                         dd.d_mass);
    Main_Update_Legacy_Neighbor_List_If_Needed(
        "Main_Refresh_Local_State", neighbor_list.FORCED_UPDATE);
    const bool sits_clustered_direct =
        selective_interaction.Has_SITS_Direct_LJ_Coulomb() &&
        !selective_interaction.Has_REST2_Direct_LJ_Coulomb();
    const int clustered_direct_solvent_numbers =
        (sits_clustered_direct || lj.Use_Clustered_Direct() ||
         lj_soft.Use_Clustered_Direct())
            ? 0
            : solvent_lj.local_solvent_numbers;
    lj.Refresh_Clustered_Metadata(clustered_direct_solvent_numbers,
                                  dd.atom_local,
                                  dd.d_excluded_list_start,
                                  dd.d_excluded_list,
                                  dd.d_excluded_numbers);
    lj_soft.Refresh_Clustered_Metadata(clustered_direct_solvent_numbers,
                                       dd.atom_local,
                                       dd.d_excluded_list_start,
                                       dd.d_excluded_list,
                                       dd.d_excluded_numbers);
    listed_forces.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                             dd.atom_local_label, dd.atom_local_id);
    if (pairwise_clustered_cache != NULL)
    {
        pairwise_clustered_cache->Refresh_Metadata(
            dd.atom_numbers, dd.atom_numbers, dd.ghost_numbers,
            dd.atom_local, dd.d_excluded_list_start, dd.d_excluded_list,
            dd.d_excluded_numbers);
    }

    angle.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                    dd.atom_local_label, dd.atom_local_id);
    urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
    bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    dihedral.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
    improper.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
    nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    restrain.Get_Local(dd.atom_local, dd.atom_numbers, dd.atom_local_label,
                       dd.atom_local_id);
    constrain.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    settle.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    vatom.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    selective_interaction.Get_Local(dd.atom_local, dd.atom_numbers,
                                    dd.ghost_numbers);
    if (selective_interaction.Uses_SITS_Listed_Forces())
    {
        sits_dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
        sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    }
    Main_Finite_Lifecycle_Probe("refresh_exit");
}

void Main_Iteration()
{
    Main_Finite_Lifecycle_Probe("iteration_entry");
    controller.Get_Time_Recorder("Iteration")->Start();
    if (md_info.need_potential || md_info.need_pressure || md_info.need_kinetic)
    {
        dd.Get_Ek_and_Temperature(&controller, &md_info);
        Main_Finite_Lifecycle_Probe("iteration_after_ek_temperature");
    }
    dd.Get_Potential(&controller, &md_info);
    Main_Finite_Lifecycle_Probe("iteration_after_potential");
    if (md_info.mode != md_info.RERUN)
    {
        Main_MC_Barostat();
        Main_Finite_Lifecycle_Probe("iteration_after_mc_barostat");
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            Main_Finite_Lifecycle_Probe(
                "iteration_before_remember_last_coordinates");
            settle.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                             md_info.pbc.rcell);
            shake.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                            md_info.pbc.rcell);
            Main_Finite_Lifecycle_Probe(
                "iteration_after_remember_last_coordinates");

            Main_Finite_Lifecycle_Probe("iteration_before_integrator");
            if (md_info.mode == md_info.NVE)
            {
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                Main_Finite_Lifecycle_Probe("iteration_after_nve");
            }
            else if (md_info.mode == md_info.MINIMIZATION)
            {
                md_info.min.Gradient_Descent(dd.atom_numbers, dd.crd, dd.frc,
                                             dd.vel, dd.d_mass_inverse);
                constrain.v_factor = fmaxf(FLT_MIN, md_info.min.momentum_keep);
                Main_Finite_Lifecycle_Probe("iteration_after_minimization");
            }
            else if (middle_langevin.is_initialized)
            {
                middle_langevin.MD_Iteration_Leap_Frog(dd.frc, dd.vel, dd.acc,
                                                       dd.crd);
                constrain.v_factor = middle_langevin.exp_gamma;
                constrain.x_factor = 0.5f * middle_langevin.exp_gamma + 0.5f;
                Main_Finite_Lifecycle_Probe("iteration_after_middle_langevin");
            }
            else if (bd_thermo.is_initialized)
            {
                bd_thermo.Record_Temperature(dd.temperature,
                                             md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bd_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
                Main_Finite_Lifecycle_Probe("iteration_after_berendsen");
            }
            else if (bussi_thermo.is_initialized)
            {
                bussi_thermo.Record_Temperature(dd.temperature,
                                                md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bussi_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
                Main_Finite_Lifecycle_Probe("iteration_after_bussi");
            }
            else if (ad_thermo.is_initialized)
            {
                if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
                {
                    ad_thermo.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc,
                                                     dd.acc, md_info.dt);
                    settle.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell);
                    shake.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell, dd.atom_numbers);
                    constrain.v_factor = FLT_MIN;
                    constrain.x_factor = 0.5;
                    Main_Finite_Lifecycle_Probe("iteration_after_andersen");
                }
                else
                {
                    md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd,
                                          dd.frc, dd.d_mass_inverse,
                                          md_info.dt);
                    constrain.v_factor = 1.0;
                    constrain.x_factor = 1.0;
                    Main_Finite_Lifecycle_Probe("iteration_after_andersen_nve");
                }
            }
            else if (nhc.is_initialized)
            {
                nhc.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc, dd.acc,
                                           md_info.dt, dd.h_ek_total,
                                           md_info.sys.freedom);
                Main_Finite_Lifecycle_Probe("iteration_after_nhc");
            }

            Main_Finite_Lifecycle_Probe("iteration_before_settle");
            settle.Do_SETTLE(dd.d_mass, dd.crd, md_info.pbc.cell,
                             md_info.pbc.rcell, dd.vel, md_info.need_pressure,
                             md_info.sys.d_stress);
            Main_Finite_Lifecycle_Probe("iteration_after_settle");
            Main_Finite_Lifecycle_Probe("iteration_before_shake");
            shake.Constrain(dd.atom_numbers, dd.crd, dd.vel, dd.d_mass_inverse,
                            dd.d_mass, md_info.pbc.cell, md_info.pbc.rcell,
                            md_info.need_pressure, md_info.sys.d_stress);
            Main_Finite_Lifecycle_Probe("iteration_after_shake");
            Main_Finite_Lifecycle_Probe("iteration_before_hard_wall");
            hard_wall.Reflect(dd.atom_numbers, dd.crd, dd.vel);
            Main_Finite_Lifecycle_Probe("iteration_after_hard_wall");
        }
        if (md_info.need_pressure && !mc_baro.is_initialized)
        {
            md_info.Get_pressure(&controller, dd.atom_numbers, dd.vel,
                                 dd.d_mass, dd.d_virial, main_stream);
            md_info.sys.Get_Density();
            press_baro.Regulate_Pressure(
                md_info.sys.steps, md_info.sys.h_stress, md_info.pbc.cell,
                md_info.dt, md_info.sys.target_pressure,
                md_info.sys.target_temperature);
            Main_Finite_Lifecycle_Probe("iteration_after_pressure");
        }
    }
    else
    {
        md_info.rerun.Iteration();
        Main_Finite_Lifecycle_Probe("iteration_after_rerun");
        if (md_info.rerun.need_box_update)
        {
            Main_Box_Change(md_info.rerun.g, 1, 0, 0);
            Main_Finite_Lifecycle_Probe("iteration_after_rerun_box_change");
        }
        md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
        Main_Finite_Lifecycle_Probe("iteration_after_rerun_global_to_dd");
    }

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        vatom.Coordinate_Refresh(dd.crd, md_info.pbc.cell, md_info.pbc.rcell);
        Main_Finite_Lifecycle_Probe("iteration_after_vatom_coord_refresh");
        if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
            md_info.mode == md_info.RERUN)
        {
            if (CONTROLLER::PP_MPI_size != 1)
            {
                controller.Get_Time_Recorder("Communication")->Start();
                dd.Exchange_Particles(&controller, &md_info);
                controller.Get_Time_Recorder("Communication")->Stop();
                Main_Finite_Lifecycle_Probe("iteration_after_exchange_particles");
                Main_Refresh_Local_State(false);
                Main_Finite_Lifecycle_Probe("iteration_after_refresh_local_state");
            }
            else
            {
                Main_Update_Legacy_Neighbor_List_If_Needed(
                    "Main_Iteration_single_rank", neighbor_list.FORCED_UPDATE);
                Main_Finite_Lifecycle_Probe(
                    "iteration_after_single_rank_neighbor_update");
            }
        }
    }
    if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
        md_info.mode == md_info.RERUN)
    {
        controller.Get_Time_Recorder("Communication")->Start();
        pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge,
                     dd.atom_numbers, dd.crd, dd.d_charge, dd.atom_local, true,
                     true, true, true);
        controller.Get_Time_Recorder("Communication")->Stop();
        Main_Finite_Lifecycle_Probe("iteration_after_pm_get_atoms");
    }
    Main_Finite_Lifecycle_Probe("iteration_exit");
    controller.Get_Time_Recorder("Iteration")->Stop();
}

void Main_Print()
{
    if (md_info.output.Check_Mdout_Step())
    {
        Main_Populate_Core_Output_Content();
        if (!md_info.pbc.pbc)
        {
            CF_NOPBC.Step_Print(&controller);
            LJ_NOPBC.Step_Print(&controller);
            gb.Step_Print(&controller);
        }
        else
        {
            lj.Step_Print(&controller);
            lj_soft.Step_Print(&controller);
            pm.Step_Print(&controller);
            selective_interaction.Step_Print(
                &controller,
                1.0f / md_info.sys.target_temperature / CONSTANT_kB);
        }
        sits_dihedral.Step_Print(&controller, false);
        sits_nb14.Step_Print(&controller, false);
        sits_cmap.Step_Print(&controller, false);

        sw.Step_Print(&controller);
        eam.Step_Print(&controller);
        tersoff.Step_Print(&controller);
        reaxff.Step_Print(&controller, md_info.d_charge);
        pairwise_force.Step_Print(&controller);
        angle.Step_Print(&controller);
        urey_bradley.Step_Print(&controller);
        bond.Step_Print(&controller);
        cmap.Step_Print(&controller);
        listed_forces.Step_Print(&controller);
        dihedral.Step_Print(&controller);
        improper.Step_Print(&controller);
        nb14.Step_Print(&controller);

        restrain.Step_Print(&controller);
        if (qc.is_initialized)
        {
            qc.Step_Print(&controller);
        }
        cv_controller.Step_Print();
        plugin.Mdout_Print();
        steer_cv.Step_Print(&controller);
        restrain_cv.Step_Print(&controller);
        meta.Step_Print(&controller);
        soft_walls.Step_Print(&controller);
        controller.Print_To_Screen_And_Mdout();
    }

    if (md_info.output.Check_Trajectory_Step())
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
        if (md_info.pbc.pbc)
        {
            md_info.mol.Molecule_Crd_Map();
            md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id, main_stream);
        }
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Vel_Traj_File();
        md_info.output.Append_Box_Traj_File();
        meta.Write_Potential();
        nhc.Save_Trajectory_File();
    }

    if (md_info.output.is_frc_traj && md_info.output.Check_Force_Step())
    {
        md_info.Frc_dd_to_Host(dd.frc, dd.atom_local_label, dd.atom_local_id,
                               main_stream);
        md_info.output.Append_Frc_Traj_File();
    }

    if (md_info.output.Check_Restart_Step())
    {
        md_info.output.Export_Restart_File();
        nhc.Save_Restart_File();
    }
}

void Main_Clear()
{
    controller.Final_Time_Summary(
        md_info.sys.steps, md_info.sys.speed_time_factor,
        md_info.sys.speed_unit_name.c_str(), md_info.mode);

    Release_Shared_LJ_Clustered_Direct_Cache();
    pairwise_clustered_cache = NULL;
    controller.Clear();
}

float Main_Box_Change(LTMatrix3 g, int scale_box, int scale_crd, int scale_vel)
{
    if (scale_box)
    {
        md_info.pbc.Update_Box(g);
    }
    // 放缩坐标与速度
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Scale_Positions_And_Velocities(
            g, scale_crd, scale_vel, dd.crd,
            dd.vel);  // rescale dd进程原子坐标与速度
        restrain.Update_Refcoord_Scaling(&md_info, g, md_info.dt, dd.atom_local,
                                         dd.atom_numbers, dd.atom_local_label,
                                         dd.atom_local_id);
    }

    // 大幅度放缩盒子时，重新初始化相关模块
    if (scale_box && md_info.pbc.Check_Change_Large())
    {
        Main_Box_Change_Largely();
    }
    else  // 更新域分解盒子
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Update_Box(g, md_info.dt);
        }
        if (CONTROLLER::PM_MPI_rank < CONTROLLER::PM_MPI_size &&
            CONTROLLER::PM_MPI_rank != -1)
        {
            pm.Update_Box(md_info.pbc.cell, md_info.pbc.rcell, g, md_info.dt);
        }
    }
    return md_info.sys.Get_Volume();
}

void Main_Box_Change_Largely()
{
    controller.printf(
        "Some modules are based on the meshing methods, and it is more "
        "precise "
        "to re-initialize these modules now for a large box change.\n");

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }
    neighbor_list.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                          md_info.nb.skin, md_info.pbc.cell, md_info.pbc.rcell);
    pm.Clear();
    pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
               md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
               md_info.no_direct_interaction_virtual_atom_numbers);
    dd.Free_Buffer();
    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Main_Refresh_Local_State(true);
        plugin.Set_Domain_Information(&dd);
    }
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);
    MPI_Barrier(MPI_COMM_WORLD);
    controller.printf(
        "------------------------------------------------------------------"
        "----"
        "--------------------------------------\n");
}

void Main_Process_Management()
{
    CONTROLLER::PM_MPI_size = pm.PM_MPI_size;
    CONTROLLER::PP_MPI_size =
        (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
         CONTROLLER::CC_MPI_size) <= 0
            ? 1
            : (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
               CONTROLLER::CC_MPI_size);

    if (CONTROLLER::MPI_size == 1)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::pm_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = 0;
        dd.pp_rank = 0;
        if (CONTROLLER::PM_MPI_size != 0)
        {
            CONTROLLER::PM_MPI_rank = 0;
            pm.pm_rank = 0;
        }
        else
        {
            CONTROLLER::PM_MPI_rank = -1;
            pm.pm_rank = -1;
        }
    }
    else if (CONTROLLER::PM_MPI_size == 0)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = CONTROLLER::MPI_rank;
        dd.pp_rank = CONTROLLER::PP_MPI_rank;
        pm.pm_rank = -1;
#ifdef USE_XCCL
        xcclUniqueId pp_id;
        if (CONTROLLER::PP_MPI_rank == 0)
        {
            xcclGetUniqueId(&pp_id);
        }
        MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
        xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size, pp_id,
                         CONTROLLER::PP_MPI_rank);
#else
        CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
    }
    else
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, CONTROLLER::MPI_rank,
                           &CONTROLLER::pp_comm);
            MPI_Comm_rank(CONTROLLER::pp_comm, &dd.pp_rank);
            CONTROLLER::PP_MPI_rank = dd.pp_rank;
#ifdef USE_XCCL
            xcclUniqueId pp_id;
            if (CONTROLLER::PP_MPI_rank == 0)
            {
                xcclGetUniqueId(&pp_id);
            }
            MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
            xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size,
                             pp_id, CONTROLLER::PP_MPI_rank);
#else
            CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
        }
        else
        {
            CONTROLLER::PP_MPI_rank =
                CONTROLLER::PP_MPI_size;  // PP_MPI_rank 设置>=
                                          // PP_MPI_size，表示非PP进程
            MPI_Comm_split(MPI_COMM_WORLD, 1, CONTROLLER::MPI_rank,
                           &CONTROLLER::pm_comm);
            MPI_Comm_rank(CONTROLLER::pm_comm, &pm.pm_rank);
            CONTROLLER::PM_MPI_rank = pm.pm_rank;
#ifdef USE_XCCL
            xcclUniqueId pm_id;
            if (CONTROLLER::PM_MPI_rank == 0)
            {
                xcclGetUniqueId(&pm_id);
            }
            MPI_Bcast(&pm_id, sizeof(pm_id), MPI_BYTE, 0, CONTROLLER::pm_comm);
            xcclCommInitRank(&CONTROLLER::d_pm_comm, CONTROLLER::PM_MPI_size,
                             pm_id, CONTROLLER::PM_MPI_rank);
#else
            CONTROLLER::d_pm_comm = CONTROLLER::pm_comm;
#endif
        }
    }

    controller.printf(
        "MPI process total: MPI_size=%d, PP_MPI_size=%d, PM_MPI_size=%d\n",
        CONTROLLER::MPI_size, CONTROLLER::PP_MPI_size, CONTROLLER::PM_MPI_size);
    controller.MPI_printf(
        "MPI process partition: MPI_rank=%d, PP_MPI_rank=%d, "
        "PM_MPI_rank=%d\n",
        CONTROLLER::MPI_rank, CONTROLLER::PP_MPI_rank, CONTROLLER::PM_MPI_rank);

    if (CONTROLLER::PP_MPI_size > 1)
    {
        md_info.nb.Excluded_List_Reform(md_info.atom_numbers);
    }
    pm.exclude_factor = CONTROLLER::PP_MPI_size == 1 ? 1.0f : 0.5f;

    deviceStreamCreate(&main_stream);
    dd.Create_Stream();
    pm.Create_Stream();

    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
}

void Main_MC_Barostat()
{
    if (mc_baro.is_initialized &&
        md_info.sys.steps % mc_baro.update_interval == 0)
    {
        mc_baro.energy_old = dd.h_sum_ene_total;
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            deviceMemcpy(mc_baro.frc_backup, dd.frc,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
            deviceMemcpy(mc_baro.crd_backup, dd.crd,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }
        mc_baro.Volume_Change_Attempt(md_info.sys.box_length, md_info.dt);
        Main_Box_Change(mc_baro.g, 1, 0, 0);
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Res_Crd_Map(mc_baro.g, md_info.dt);
        }

        Main_Calculate_Force();
        dd.Get_Potential(&controller, &md_info);
        mc_baro.energy_new = dd.h_sum_ene_total;
        mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV -
                             md_info.ug.ug_numbers * CONSTANT_kB *
                                 md_info.sys.target_temperature *
                                 logf(mc_baro.VDevided);
        if (mc_baro.couple_dimension != mc_baro.NO &&
            mc_baro.couple_dimension != mc_baro.XYZ)
        {
            mc_baro.extra_term -= mc_baro.surface_number *
                                  mc_baro.surface_tension * mc_baro.DeltaS;
        }
        mc_baro.accept_possibility =
            mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility =
            expf(-mc_baro.accept_possibility /
                 (CONSTANT_kB * md_info.sys.target_temperature));

        if (!mc_baro.Check_MC_Barostat_Accept())  // 如果不接受
        {
            mc_baro.g = {-mc_baro.g.a11, 0, -mc_baro.g.a22, 0, 0,
                         -mc_baro.g.a33};
            if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
            {
                deviceMemcpy(dd.frc, mc_baro.frc_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
                deviceMemcpy(dd.crd, mc_baro.crd_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
            }
            Main_Box_Change(mc_baro.g, 1, 0, 0);
        }
        mc_baro.Delta_Box_Length_Max_Update();
        dd.h_sum_ene_total = mc_baro.energy_old;  // 恢复能量值
    }
}

void Main_Sync_Dynamic_Targets_To_Controllers()
{
    md_info.sys.Update_Targets_By_Schedule(md_info.sys.steps);
    const float target_temperature = md_info.sys.target_temperature;
    bd_thermo.Set_Target_Temperature(target_temperature);
    bussi_thermo.Set_Target_Temperature(target_temperature);
    ad_thermo.Set_Target_Temperature(target_temperature);
    middle_langevin.Set_Target_Temperature(target_temperature);
    nhc.Set_Target_Temperature(target_temperature);
}
