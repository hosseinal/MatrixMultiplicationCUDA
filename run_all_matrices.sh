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

# Use relative locations from script start directory (no pwd/ROOT_DIR required)
# This assumes the script is started from the project root (where this file lives)
BIN="./build/matrix_multiplication"
MODE="normal"  # normal | nvbench
MATRICES_DIR="./matrices"

# # Always build the project before running benchmarks. This runs CMake and builds
# # the `matrix_multiplication` target in the `build/` directory.
# echo "Building project (cmake -> build)..."
# cd "$ROOT_DIR"
# echo "$ROOT_DIR"
# mkdir build
# # configure
# cd build
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make
# cd ..



if [[ ! -d "$MATRICES_DIR" ]]; then
  echo "ERROR: matrices directory not found: $MATRICES_DIR"
  exit 1
fi

# CSV header for combined results
header="name,time_ms,diff,maxrelativeerr,size,pattern,sparsity,matrixApath"

# optional first arg: pattern to filter (e.g. ./run_all_matrices.sh random)
# optional second arg: mode to run: "normal" (default) or "nvbench"
PATTERN_FILTER=""
if [[ ${#@} -ge 1 ]]; then
  PATTERN_FILTER="$1"
  echo "Pattern filter enabled: $PATTERN_FILTER"
fi
if [[ ${#@} -ge 2 ]]; then
  MODE="$2"
  echo "Mode: $MODE"
fi

# select binary based on mode
if [[ "$MODE" == "nvbench" ]]; then
  BIN="./build/matrix_multiplication_nvbench"
else
  BIN="./build/matrix_multiplication"
fi

# verify selected binary exists
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found or not executable: $BIN"
  echo "Build the project first (cd build && make) or pass the correct mode/binary"
  exit 1
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

    echo "Running: A=$a B=$b sparsity=${sparsity} (mode=${MODE})"
    # per-pattern result file (under the pattern directory)
    PATTERN_OUT="$pattern_dir${pattern}_results.csv"
    mkdir -p "$pattern_dir"
    if [[ ! -f "$PATTERN_OUT" ]]; then
      echo "$header" > "$PATTERN_OUT"
    fi

    if [[ "$MODE" == "nvbench" ]]; then
      # nvbench executable reads input paths from env vars (MATRIX_A_PATH, MATRIX_B_PATH)
      # force CSV output and append single invocation output to files
      NV_OUTFILE="$pattern_dir${pattern}_nvbench.csv"
      if [[ ! -f "$NV_OUTFILE" ]]; then
        echo "$header" > "$NV_OUTFILE"
      fi
  # single invocation: write CSV output and append to pattern results, pattern nvbench file, and the combined file
  MATRIX_A_PATH="$a" MATRIX_B_PATH="$b" NV_OUTPUT_FORMAT=csv "$BIN" | tee -a "$PATTERN_OUT" "$NV_OUTFILE" >> "$ALL_OUT"
    else
      # normal executable uses argv: A B pattern sparsity
      "$BIN" "$a" "$b" "$pattern" "$sparsity" >> "$PATTERN_OUT"
      # also copy line to combined file
      tail -n 1 "$PATTERN_OUT" >> "$ALL_OUT"
    fi
  done
done

echo "Wrote combined results -> $ALL_OUT"

echo "All done."
