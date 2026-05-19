from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("rest2-perf")
    group.addoption(
        "--rest2-perf-sponge-cmd",
        action="store",
        default=None,
        help="Path to the SPONGE executable used by REST2 benchmarks.",
    )
    group.addoption(
        "--rest2-perf-manager-cmd",
        action="store",
        default=None,
        help="Path to SPONGE_MANAGER used by REST2-REMD benchmarks.",
    )
    group.addoption(
        "--rest2-perf-steps",
        action="store",
        type=int,
        default=100,
        help="Step count for each REST2 performance micro-benchmark.",
    )
    group.addoption(
        "--rest2-perf-timeout",
        action="store",
        type=int,
        default=1200,
        help="Timeout in seconds for each REST2 performance command.",
    )
    group.addoption(
        "--fep-rest2-root",
        action="store",
        default=None,
        help=(
            "Root directory of the four-replica FEP test system used by the "
            "FEP+REST2 manager benchmark. Defaults to the repository fixture."
        ),
    )


@pytest.fixture(scope="session")
def sponge_cmd(pytestconfig):
    value = pytestconfig.getoption("--rest2-perf-sponge-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def manager_cmd(pytestconfig):
    value = pytestconfig.getoption("--rest2-perf-manager-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def rest2_perf_steps(pytestconfig):
    value = int(pytestconfig.getoption("--rest2-perf-steps"))
    if value <= 0:
        raise pytest.UsageError("--rest2-perf-steps must be positive")
    return value


@pytest.fixture(scope="session")
def rest2_perf_timeout(pytestconfig):
    value = int(pytestconfig.getoption("--rest2-perf-timeout"))
    if value <= 0:
        raise pytest.UsageError("--rest2-perf-timeout must be positive")
    return value


@pytest.fixture(scope="session")
def fep_rest2_root(pytestconfig):
    value = pytestconfig.getoption("--fep-rest2-root")
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[1] / "data" / "fep_test_for_remd"
