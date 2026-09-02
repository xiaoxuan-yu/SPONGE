import os
import shutil
from pathlib import Path


def repo_root_from_test_file(test_file):
    return Path(test_file).resolve().parents[4]


def resolve_executable(explicit_path, repo_root):
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Executable not found: {explicit_path}")
        return explicit_path

    found = shutil.which("SPONGE")
    if found:
        return Path(found).resolve()

    for build_dir in (
        "build-dev-cuda13-hostcc",
        "build-dev-cuda13",
        "build-dev-cuda12",
        "build",
    ):
        candidate = repo_root / build_dir / "SPONGE"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError("Cannot find SPONGE; pass --sponge-cmd")


def runtime_env(repo_root):
    env = os.environ.copy()
    pixi_env = repo_root / ".pixi" / "envs" / "dev-cuda13"
    library_paths = [
        pixi_env / "lib",
        pixi_env / "targets" / "x86_64-linux" / "lib",
    ]
    existing = env.get("LD_LIBRARY_PATH")
    path_text = os.pathsep.join(
        str(path) for path in library_paths if path.exists()
    )
    if path_text:
        env["LD_LIBRARY_PATH"] = (
            path_text if not existing else path_text + os.pathsep + existing
        )
    return env


def copy_ala2_case(repo_root, outputs_path, run_name):
    source = repo_root / "benchmarks/performance/sits/statics/ala2_sits"
    if not source.exists():
        raise FileNotFoundError(f"ALA2 fixture not found: {source}")
    case_dir = Path(outputs_path) / run_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, case_dir)
    return case_dir
