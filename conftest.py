"""Project-wide pytest configuration."""

import warnings

from numba.core.errors import NumbaPerformanceWarning
# numba is still used in femcubes.core.utils.maths_utils; this filter
# suppresses the "faster on contiguous arrays" noise it emits during tests.


def _suppress_numba_matmul_warning() -> None:
    """Apply the targeted Numba matmul performance-warning filter."""
    warnings.filterwarnings(
        "ignore",
        message=r".*faster on contiguous arrays.*",
        category=NumbaPerformanceWarning,
    )


def pytest_configure(config):
    """Apply project-wide warning filters once pytest is configured."""
    _suppress_numba_matmul_warning()


def pytest_runtest_setup(item):
    """Re-apply warning filters after pytest resets per-test warning state."""
    _suppress_numba_matmul_warning()


def pytest_collection_modifyitems(session, config, items):
    """Schedule long full-model regressions before shorter unit-style tests."""
    items.sort(
        key=lambda item: 0 if item.get_closest_marker("full_model") is not None else 1
    )
