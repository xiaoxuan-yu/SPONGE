from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("rest2")
    group.addoption(
        "--sponge-cmd",
        action="store",
        default=None,
        help="Path to the SPONGE executable used by REST2 validation tests.",
    )
    group.addoption(
        "--manager-cmd",
        action="store",
        default=None,
        help=(
            "Path to SPONGE_MANAGER. Defaults to PATH or common build "
            "directories."
        ),
    )
    group.addoption(
        "--rest2-timeout",
        action="store",
        type=int,
        default=600,
        help="Timeout in seconds for each REST2 validation command.",
    )


@pytest.fixture(scope="session")
def sponge_cmd(pytestconfig):
    value = pytestconfig.getoption("--sponge-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def manager_cmd(pytestconfig):
    value = pytestconfig.getoption("--manager-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def rest2_timeout(pytestconfig):
    value = int(pytestconfig.getoption("--rest2-timeout"))
    if value <= 0:
        raise pytest.UsageError("--rest2-timeout must be positive")
    return value
