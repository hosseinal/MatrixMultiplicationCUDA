#!/usr/bin/env bash
set -euo pipefail

# generate_all_matrices.sh
# Creates a `matrices/` directory and generates matrices for multiple
# patterns, sizes and sparsity levels using matrix_generator.py.
# Usage: ./generate_all_matrices.sh [-y]

FORCE=0
if [[ ${1:-} == "-y" || ${1:-} == "--yes" ]]; then
  FORCE=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$ROOT_DIR/matrix_generator.py"
OUT_DIR="$ROOT_DIR/matrices"

if [[ ! -f "$GENERATOR" ]]; then
  echo "ERROR: matrix_generator.py not found at: $GENERATOR"
  exit 1
fi

patterns=(random checkerboard diagonal blockdiagonal blockrandom)
# sizes (square matrices)
sizes=(128 256 512 1024 2048 4096 8192 16384)
# sparsity values
sparsities=(0.5 0.6 0.7 0.8 0.9)

echo "This script will create many matrix files under: $OUT_DIR"
if [[ $FORCE -eq 0 ]]; then
  echo "Patterns: ${patterns[*]}"
  echo "Sizes: ${sizes[*]}"
  echo "Sparsities: ${sparsities[*]}"
  echo "Estimated number of matrix files: $(( ${#patterns[@]} * ${#sizes[@]} * ${#sparsities[@]} * 2 )) (A and B each)"
  echo "Note: Large sizes (>= 8192) may produce very large files and use a lot of disk space and memory."
  read -p "Continue? [y/N] " yn
  case "$yn" in
    [Yy]*) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

mkdir -p "$OUT_DIR"

for pattern in "${patterns[@]}"; do
  pattern_dir="$OUT_DIR/$pattern"
  mkdir -p "$pattern_dir"
  echo "Generating pattern: $pattern -> $pattern_dir"
  for size in "${sizes[@]}"; do
    for sp in "${sparsities[@]}"; do
      outA="$pattern_dir/A_${size}_s${sp}.mat"
      outB="$pattern_dir/B_${size}_s${sp}.mat"
      echo "  Generating A: size=$size sparsity=$sp -> $outA"
      python3 "$GENERATOR" -o "$outA" -n "$size" -t float16 -p "$pattern" -b 16
      echo "  Generating B: size=$size sparsity=$sp -> $outB"
      python3 "$GENERATOR" -o "$outB" -n "$size" -t float16 -p "$pattern" -b 16
    done
  done
done

echo "All matrices generated under: $OUT_DIR"
