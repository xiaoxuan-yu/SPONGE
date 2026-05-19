import json

from benchmarks.performance.remd.tests.utils import (
    print_summary_table,
    repo_root_from_test_file,
    resolve_executable,
    run_ice_tremd_case,
)


def test_ice_ih_tremd_worker_transport_perf(
    outputs_path,
    ice_root,
    remd_temperatures,
    remd_block_steps,
    remd_epochs,
    remd_worker_modes,
    manager_cmd,
    sponge_cmd,
    remd_timeout,
):
    repo_root = repo_root_from_test_file(__file__)
    resolved_manager = resolve_executable(
        manager_cmd, "SPONGE_MANAGER", repo_root
    )
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)

    case_root = outputs_path / "ice_ih_tremd"
    case_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for mode in remd_worker_modes:
        run_dir = case_root / mode
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        summary = run_ice_tremd_case(
            run_dir=run_dir,
            mode=mode,
            ice_root=ice_root,
            temperatures=remd_temperatures,
            block_steps=remd_block_steps,
            epochs=remd_epochs,
            manager_cmd=resolved_manager,
            sponge_cmd=resolved_sponge,
            timeout=remd_timeout,
            repo_root=repo_root,
        )
        summaries.append(summary)

        assert summary["exchange_attempts"] > 0
        assert 0.0 <= summary["acceptance_ratio"] <= 1.0
        assert summary["aggregate_steps_per_s"] > 0.0
        assert sorted(summary["final_walker_ids"]) == list(
            range(len(remd_temperatures))
        )

    (case_root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary_table(summaries)
