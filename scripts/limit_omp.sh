#!/bin/bash
# Limit OpenMP and BLAS to one thread per process, so that MPI ranks and
# parallel test workers do not oversubscribe the machine.
#
# Usage:
#   source scripts/limit_omp.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
