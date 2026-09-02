from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("selective-interaction")
    group.addoption(
        "--sponge-cmd",
        action="store",
        default=None,
        help="Path to the SPONGE executable used by selective-interaction tests.",
    )
    group.addoption(
        "--selective-timeout",
        action="store",
        type=int,
        default=600,
        help="Timeout in seconds for each selective-interaction command.",
    )


@pytest.fixture(scope="session")
def sponge_cmd(pytestconfig):
    value = pytestconfig.getoption("--sponge-cmd")
    return Path(value).resolve() if value else None


@pytest.fixture(scope="session")
def selective_timeout(pytestconfig):
    value = int(pytestconfig.getoption("--selective-timeout"))
    if value <= 0:
        raise pytest.UsageError("--selective-timeout must be positive")
    return value
