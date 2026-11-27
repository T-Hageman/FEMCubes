cmake_minimum_required(VERSION 3.18)
project(femcubes_cpp_kernels CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with aggressive optimisation by default.
# Note: -ffast-math is intentionally omitted — it breaks strict IEEE 754
# arithmetic and causes dF_dl + dE_dl to evaluate to a tiny negative value
# at the exact yield surface, triggering the fracture branch incorrectly
# and preventing Newton convergence.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(WIN32)
  message(FATAL_ERROR
    "Windows is not supported. FEniCSx (a required dependency of FEMCubes) "
    "does not run natively on Windows. Please use WSL2 (Windows Subsystem "
    "for Linux) with a Ubuntu distribution and follow the Linux install "
    "instructions instead.")
endif()
add_compile_options(-O3)

# -march=native is OFF by default so that compiled extensions are portable.
# A developer building locally for maximum performance can enable it:
#   cmake -DFEMCUBES_NATIVE_ARCH=ON ...
option(FEMCUBES_NATIVE_ARCH "Optimise for the build machine's CPU" OFF)
if(FEMCUBES_NATIVE_ARCH)
  add_compile_options(-march=native)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# pybind11 – prefer the cmake config installed with the Python package
find_package(pybind11 CONFIG REQUIRED)

# Eigen3
# Try the system/conda install first; if not found, download via FetchContent.
# Eigen 5 treats a version request of 3.3 as the 3.3 series, not a minimum.
find_package(Eigen3 QUIET NO_MODULE)
if(Eigen3_FOUND AND Eigen3_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Eigen 3.3 or newer is required (found ${Eigen3_VERSION}).")
endif()
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally — downloading via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# ---------------------------------------------------------------------------
# Visco-elastic J2 return-mapping kernel
# ---------------------------------------------------------------------------

set(VE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/viscoelastic)

pybind11_add_module(_viscoelastic_kernels_cpp
    ${VE_SRC}/viscoelastic_kernels_internal.cpp
    ${VE_SRC}/split_viscoelastic.cpp
    ${VE_SRC}/split_viscoelastic_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared/phasefield_splits_pybind.cpp
    ${VE_SRC}/viscoelastic_kernels_pybind.cpp
)

target_include_directories(_viscoelastic_kernels_cpp PRIVATE
    ${VE_SRC}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(_viscoelastic_kernels_cpp PRIVATE Eigen3::Eigen)



install(TARGETS _viscoelastic_kernels_cpp
    LIBRARY DESTINATION femcubes/solids/cpp_kernels/viscoelastic
)

# ---------------------------------------------------------------------------
# Combined cohesive–visco-elastic kernel
# ---------------------------------------------------------------------------

set(CVE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cohesive_viscoelastic)

pybind11_add_module(_cohesive_viscoelastic_kernels_cpp
    ${CVE_SRC}/cohesive_viscoelastic_kernels_internal.cpp
    ${CVE_SRC}/cohesive_viscoelastic_kernels_pybind.cpp
)

target_include_directories(_cohesive_viscoelastic_kernels_cpp PRIVATE
    ${CVE_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/shared
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(_cohesive_viscoelastic_kernels_cpp PRIVATE Eigen3::Eigen)

install(TARGETS _cohesive_viscoelastic_kernels_cpp
    LIBRARY DESTINATION femcubes/solids/cpp_kernels/cohesive_viscoelastic
)
