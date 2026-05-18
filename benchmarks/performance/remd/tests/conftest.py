from pathlib import Path

import pytest


def _parse_temperatures(value):
    try:
        temperatures = [float(item) for item in str(value).split(",") if item]
    except ValueError as exc:
        raise pytest.UsageError(
            "--temperatures must be a comma-separated list"
        ) from exc
    if len(temperatures) < 2:
        raise pytest.UsageError(
            "--temperatures must contain at least two values"
        )
    if any(temperature <= 0.0 for temperature in temperatures):
        raise pytest.UsageError("--temperatures must be positive")
    return temperatures


def _parse_modes(value):
    allowed = {"file", "tcp", "shm"}
    modes = [item.strip() for item in str(value).split(",") if item.strip()]
    if not modes:
        raise pytest.UsageError("--remd-worker-modes must not be empty")
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise pytest.UsageError(
            "--remd-worker-modes contains unsupported mode(s): "
            + ", ".join(invalid)
        )
    return modes


def pytest_addoption(parser):
    group = parser.getgroup("remd")
    group.addoption(
        "--ice-root",
        action="store",
        default="/mnt/data8t/Data/ice_Ih_cubic_box",
        help="Root directory of the pre-equilibrated ice Ih benchmark case.",
    )
    group.addoption(
        "--temperatures",
        action="store",
        default="100,110,120,130,140",
        help="Comma-separated T-REMD temperature ladder in Kelvin.",
    )
    group.addoption(
        "--block-steps",
        action="store",
        type=int,
        default=5,
        help="MD steps per manager block.",
    )
    group.addoption(
        "--epochs",
        action="store",
        type=int,
        default=3,
        help="Number of manager epochs.",
    )
    group.addoption(
        "--remd-worker-modes",
        action="store",
        default="tcp,shm",
        help=("Comma-separated worker modes to benchmark: file,tcp,shm."),
    )
    group.addoption(
        "--manager-cmd",
        action="store",
        default=None,
        help="Path to SPONGE_MANAGER. Defaults to PATH or common build dirs.",
    )
    group.addoption(
        "--sponge-cmd",
        action="store",
        default=None,
        help="Path to SPONGE worker executable.",
    )
    group.addoption(
        "--remd-timeout",
        action="store",
        type=int,
        default=7200,
        help="Timeout in seconds for each REMD benchmark mode.",
    )


@pytest.fixture(scope="session")
def ice_root(pytestconfig):
    path = Path(pytestconfig.getoption("--ice-root")).resolve()
    if not path.exists():
        raise pytest.UsageError(f"--ice-root does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def remd_temperatures(pytestconfig):
    return _parse_temperatures(pytestconfig.getoption("--temperatures"))


@pytest.fixture(scope="session")
def remd_block_steps(pytestconfig):
    value = int(pytestconfig.getoption("--block-steps"))
    if value <= 0:
        raise pytest.UsageError("--block-steps must be positive")
    return value


@pytest.fixture(scope="session")
def remd_epochs(pytestconfig):
    value = int(pytestconfig.getoption("--epochs"))
    if value <= 0:
        raise pytest.UsageError("--epochs must be positive")
    return value


@pytest.fixture(scope="session")
def remd_worker_modes(pytestconfig):
    return _parse_modes(pytestconfig.getoption("--remd-worker-modes"))


@pytest.fixture(scope="session")
def manager_cmd(pytestconfig):
    value = pytestconfig.getoption("--manager-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def sponge_cmd(pytestconfig):
    value = pytestconfig.getoption("--sponge-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def remd_timeout(pytestconfig):
    return int(pytestconfig.getoption("--remd-timeout"))
