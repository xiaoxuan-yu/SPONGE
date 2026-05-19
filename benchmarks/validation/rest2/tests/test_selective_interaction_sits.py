from benchmarks.performance.sits.tests.utils import write_sits_mdin
from benchmarks.utils import Runner
from benchmarks.validation.rest2.tests.utils import (
    copy_ala2_case,
    repo_root_from_test_file,
    resolve_executable,
    runtime_env,
)
from benchmarks.validation.utils import parse_mdout_rows


def test_sits_still_runs_through_selective_interaction_facade(
    outputs_path, sponge_cmd, rest2_timeout
):
    repo_root = repo_root_from_test_file(__file__)
    case_dir = copy_ala2_case(repo_root, outputs_path, "sits_facade_smoke")
    write_sits_mdin(
        case_dir,
        step_limit=20,
        dt=0.002,
        cutoff=8.0,
        thermostat_tau=1.0,
        write_information_interval=10,
        write_mdout_interval=5,
        default_in_file_prefix="ALA",
        sits_mode="iteration",
        sits_atom_numbers=22,
        sits_k_numbers=4,
        sits_t_low=273.0,
        sits_t_high=650.0,
        sits_record_interval=1,
        sits_update_interval=10,
        sits_pe_a=1.0,
        sits_pe_b=34.23,
        constrain_mode="SHAKE",
    )

    output = Runner.run_sponge(
        case_dir,
        timeout=rest2_timeout,
        sponge_cmd=resolve_executable(sponge_cmd, "SPONGE", repo_root),
        env=runtime_env(repo_root),
    )
    assert "SITS mode = iteration" in output
    assert (case_dir / "SITS_nk_rest.txt").exists()

    rows = parse_mdout_rows(
        case_dir / "mdout.txt",
        ("SITS_bias", "SITS_fb", "SITS_AA_kAB"),
        int_columns=(),
    )
    assert rows
