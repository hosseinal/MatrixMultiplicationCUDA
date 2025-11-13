#!/usr/bin/env bash
set -euo pipefail

# generate_block_matrices.sh
# Generates a series of block-random matrices using matrix_generator.py
# Default block size: 32 (rows) x 16 (cols)
# Default sparsity sweep: 60 -> 90 (%) in STEP percent increments (default 5)

# USAGE:
#   ./generate_block_matrices.sh [START] [END] [STEP]
# Example: ./generate_block_matrices.sh 60 90 5
# If no args given, defaults to 60 90 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/matrix_generator.py"
OUTDIR="$SCRIPT_DIR/generated_matrices"
mkdir -p "$OUTDIR"

# Default sparsity sweep (percent)
START=${1:-60}
END=${2:-90}
STEP=${3:-5}

# Block size as requested: 32 rows x 16 cols
BLOCK_ROWS=32
BLOCK_COLS=16

# List of desired matrix sizes (rows x cols)
SIZES=(
  "1024x256"
  "256x2304"
  "512x2048"
  "128x1152"
  "256x64"
  "512x128"
  "512x4608"
  "256x1024"
  "1024x512"
  "1000x2048"
  "64x256"
  "128x256"
  "128x512"
  "64x147"
  "512x256"
  "2048x512"
  "512x1024"
  "256x512"
  "2048x1024"
  "512x512"
)

# Validate python exists
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH. Please install Python 3 and try again." >&2
  exit 2
fi

echo "Generating matrices into: $OUTDIR"
echo "Block size: ${BLOCK_ROWS}x${BLOCK_COLS}, sparsity sweep: ${START}% -> ${END}% step ${STEP}%"

for size in "${SIZES[@]}"; do
  nrows=${size%x*}
  ncols=${size#*x}
  for s in $(seq "$START" "$STEP" "$END"); do
    # Convert percent to fraction (e.g. 60 -> 0.60)
    s_frac=$(python3 - <<PY
print(float($s)/100.0)
PY
)
  out_file="$OUTDIR/m_${nrows}x${ncols}_sp${s}.mtx"
    echo "Generating ${nrows}x${ncols} sparsity ${s}% -> ${out_file}"
    python3 "$GENERATOR" -n "$nrows" -m "$ncols" -s "$s_frac" -t float32 -p blockrandom -br "$BLOCK_ROWS" -bc "$BLOCK_COLS" -o "$out_file"
  done
done

echo "Done. Generated files are in: $OUTDIR" 
