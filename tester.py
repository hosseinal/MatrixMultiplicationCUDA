import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import re

from matrix_generator import generate_matrix


MATRIX_SIZES = [0x500, 0x1000, 0x1500, 0x2000]
TEST_PATTERNS = ["blockrandom"]
SPARSITY = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_RUNS = 30
GENERATE_TEST_FILES = True
RUN_TESTS = True

for size in MATRIX_SIZES:
    for run in range(NUM_RUNS):
        matrix_b = f"tests/MatrixB_{size}_{run}.mat"

        if GENERATE_TEST_FILES:
            generate_matrix(output=matrix_b,
                            num_cols=size,
                            num_rows=size,
                            sparsity=0.0,
                            dtype='float16',
                            heatmap=False,
                            pattern='random',
                            blocksize=16)

            print(f"INFO: Created {matrix_b}")

        for pattern in TEST_PATTERNS:
            for sparsity in SPARSITY:
                matrix_a = f"tests/MatrixA_{size}_{pattern}_{sparsity}_{run}.mat"

                if GENERATE_TEST_FILES:
                    generate_matrix(output=matrix_a,
                                    num_cols=size,
                                    num_rows=size,
                                    sparsity=sparsity,
                                    dtype='float16',
                                    heatmap=False,
                                    pattern=pattern,
                                    blocksize=16)

                    print(f"INFO: Created {matrix_a}")

                if RUN_TESTS:
                    print(f"INFO: Running {matrix_a} x {matrix_b}")

                    result = subprocess.run(
                        ["./cmake-build-debug/matrix_multiplication",
                             matrix_a, matrix_b], stdout=subprocess.PIPE)

                    print(f"INFO: Finished running {matrix_a} x {matrix_b}")

                    matrix_c = f"tests/MatrixC_{size}_{pattern}_{sparsity}_{run}.mat"

                    with open(matrix_c, "w+") as f:
                        f.write(result.stdout.decode("utf-8"))

print("INFO: Scrapping results")

test_results = [str(p) for p in Path('tests/').rglob("MatrixC*")]

data = {'run':[], 'time':[], 'size':[], 'pattern':[], 'sparsity':[],
        'algorithm': [], 'gflops': [], 'rel_error': []}

for file in test_results:
    pattern = r'tests/MatrixC_(\d+)_([^_]+)_(\d+\.\d+)_(\d+)\.mat'
    match = re.match(pattern, file)

    if not match:
        print(f"no match for {file}")
        break

    matrix_size = int(match.group(1))
    matrix_pattern = match.group(2)
    sparsity = float(match.group(3))
    run_number = int(match.group(4))

    with open(file) as f:
        lines = f.readlines()

    for line in range(3, len(lines), 5):
        if lines[line].strip().startswith('CUDA'):
            line += 1
        algo = lines[line].split("time")[0].strip()
        time = float(lines[line].split(":")[1].strip())
        rel_error = float(lines[line+3].split(":")[1].strip())
        #(2 × N³ - N²) / (time_in_ms × 10⁶)

        if time != 0:
            gflops = (2.0 * matrix_size ** 3) / (time * 10 ** 6)
        else:
            gflops = -1

        data['run'].append(run_number)
        data['time'].append(time)
        data['size'].append(matrix_size)
        data['pattern'].append(matrix_pattern)

        data['sparsity'].append(sparsity)
        data['algorithm'].append(algo)
        data['gflops'].append(gflops)
        data['rel_error'].append(rel_error)

df = pd.DataFrame(data)
df.loc[df['algorithm'] == 'Dense on GPU', 'rel_error'] = 0.0
df.loc[df['gflops'] == -1, 'gflops'] = np.nan
df[df['algorithm'] == 'Dense on GPU'].head()
algorithms = df['algorithm'].unique().tolist()

df.to_csv('results.csv')

print("INFO: results.csv created")
