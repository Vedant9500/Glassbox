#!/usr/bin/env bash
# Safe parallel test runner.
#
# Caps per-process math-library threads so total CPU use stays ~14 threads
# (7 workers x 2): without these caps each pytest worker's C++/torch code can
# spawn up to nproc OpenMP threads and oversubscribe/lock up the machine.
#
# Usage: scripts/test_fast.sh [extra pytest args...]
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
exec python3 -m pytest tests/ -q --tb=line -n "${TEST_WORKERS:-7}" "$@"
