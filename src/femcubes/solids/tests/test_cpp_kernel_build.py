"""Build configuration checks using the development environment's dependencies."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_cpp_kernels_configure_with_installed_eigen_without_downloads(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to configure the C++ kernels.")
    pybind11 = pytest.importorskip("pybind11")
    source_dir = Path(__file__).resolve().parents[1] / "cpp_kernels"

    # A system Eigen must not mask rejection of the environment's Eigen package.
    result = subprocess.run(
        [
            cmake,
            "-S", str(source_dir),
            "-B", str(tmp_path / "build"),
            f"-DCMAKE_PREFIX_PATH={sys.prefix}",
            f"-DCMAKE_FIND_ROOT_PATH={sys.prefix}",
            "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
