#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 10:00:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


set -euo pipefail

# run_all_matrices.sh
# For each pattern folder under matrices/, runs the compiled binary
# `build/matrix_multiplication` for each A/B pair and stores output into
# `matrices/<pattern>/results.csv`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/build/matrix_multiplication"
MATRICES_DIR="$ROOT_DIR/matrices"

# Always build the project before running benchmarks. This runs CMake and builds
# the `matrix_multiplication` target in the `build/` directory.
echo "Building project (cmake -> build)..."
cd "$ROOT_DIR"
mkdir build
# configure
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..


if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found or not executable: $BIN"
  echo "Build the project first (cd build && make)"
  exit 1
fi

if [[ ! -d "$MATRICES_DIR" ]]; then
  echo "ERROR: matrices directory not found: $MATRICES_DIR"
  exit 1
fi

# CSV header for combined results
header="name,time_ms,diff,maxrelativeerr,size,pattern,sparsity,matrixApath"

# optional first arg: pattern to filter (e.g. ./run_all_matrices.sh random)
PATTERN_FILTER=""
if [[ ${#@} -ge 1 ]]; then
  PATTERN_FILTER="$1"
  echo "Pattern filter enabled: $PATTERN_FILTER"
fi

ALL_OUT="$MATRICES_DIR/all_results.csv"
echo "writing combined results to: $ALL_OUT"
echo "$header" > "$ALL_OUT"

for pattern_dir in "$MATRICES_DIR"/*/; do
  pattern=$(basename "$pattern_dir")
  if [[ -n "$PATTERN_FILTER" && "$pattern" != "$PATTERN_FILTER" ]]; then
    continue
  fi

  # find A_*.mat and B_*.mat pairs by size/sparsity
  # assume files named as A_<size>_s<sp>.mat and B_<size>_s<sp>.mat
  for a in "$pattern_dir"/A_*.mat; do
    [[ -e "$a" ]] || continue
    fname=$(basename "$a")
    # derive expected B file
    b="$pattern_dir/${fname/A_/B_}"
    if [[ ! -f "$b" ]]; then
      echo "Skipping $a: matching B not found"
      continue
    fi
    # extract sparsity from filename A_<size>_s<sp>.mat -> sp (e.g. 0.5)
    sparsity=""
    if [[ "$fname" =~ _s([0-9]+\.?[0-9]*) ]]; then
      sparsity="${BASH_REMATCH[1]}"
    fi

    echo "Running: A=$a B=$b sparsity=${sparsity}"
    # pass pattern as third arg and sparsity as fourth so the program prints it
    # capture program output (CSV lines) and append to combined ALL_OUT
    "$BIN" "$a" "$b" "$pattern" "$sparsity" >> "$ALL_OUT"
  done
done

echo "Wrote combined results -> $ALL_OUT"

echo "All done."
