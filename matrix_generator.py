# !/usr/bin/python3

import argparse
import numpy as np
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt

def generate_dense_matrix(block_shape, dtype):
    if dtype.startswith('int'):
        return np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max,
                                 block_shape).astype(dtype)
    else:
        return np.random.random(block_shape).astype(dtype)


def generate_matrix(output: str, num_rows: int, num_cols: int, sparsity: float,
                    dtype: str, heatmap: bool = False, pattern: str = 'random',
                    blocksize: int = 16):

    matrix = np.zeros((num_rows, num_cols), dtype=dtype)

    def fill_block(block_i, block_j):
        # fill the matrix from i to i + blocksize and j to j + blocksize
        start_row = block_i * blocksize
        start_col = block_j * blocksize
        end_row = min(start_row + blocksize, num_rows)
        end_col = min(start_col + blocksize, num_cols)

        block_shape = (end_row - start_row, end_col - start_col)

        matrix[start_row:end_row, start_col:end_col] = generate_dense_matrix(
            block_shape, dtype)

    if pattern == 'random':
        matrix = sparse.random(num_rows, num_cols, (1.0 - sparsity),
                               dtype=dtype if dtype != 'float16' else 'float32'
                               ).toarray().astype(dtype)
    elif pattern == 'checkerboard':
        for i in range(num_rows // blocksize + 1):
            for j in range(num_cols // blocksize + 1):
                if i % 2 ^ j % 2 == 0:
                    fill_block(i, j)

    elif pattern == 'diagonal':
        for i in range(min(num_rows, num_cols)):
            matrix[i:i+1, i:i+1] = generate_dense_matrix((1,1), dtype)

    elif pattern == 'blockdiagonal':
        for i in range(min(num_rows, num_cols) // blocksize + 1):
            fill_block(i, i)

    elif pattern == 'blockrandom':
        mask = sparse.random(num_rows // blocksize + 1,
                             num_cols // blocksize + 1,
                             (1.0 - sparsity), dtype='int32').toarray()
        for i in range(num_rows // blocksize + 1):
            for j in range(num_cols // blocksize + 1):
                if mask[i][j] != 0:
                    fill_block(i, j)

    elif pattern in ('largerandom', 'largerandom_block', 'largerandom block'):
        # Generate rectangular block-random pattern where blocks are 64 rows x 16 cols
        brow, bcol = 64, 16
        mask = sparse.random(num_rows // brow + 1,
                             num_cols // bcol + 1,
                             (1.0 - sparsity), dtype='int32').toarray()
        for i in range(num_rows // brow + 1):
            for j in range(num_cols // bcol + 1):
                if mask[i][j] != 0:
                    start_row = i * brow
                    start_col = j * bcol
                    end_row = min(start_row + brow, num_rows)
                    end_col = min(start_col + bcol, num_cols)
                    block_shape = (end_row - start_row, end_col - start_col)
                    matrix[start_row:end_row, start_col:end_col] = generate_dense_matrix(block_shape, dtype)


    else:
        print("Pattern not recognized")
        exit(1)

    matrix_str = ""

    for i in range(num_rows):
        for j in range(num_cols):
            matrix_str += str(matrix[i][j]) + ' '
        matrix_str += '\n'

    if output is not None:
        with open(output, 'w') as f:
            f.write(f"{num_rows} {num_cols}\n")
            f.write(matrix_str)
    else:
        print(matrix_str)

    if heatmap is not None and heatmap:
        sns.heatmap(matrix, cbar=True, cmap=plt.get_cmap('jet'))
        plt.title("Matrix heatmap")

        if output is not None:
            plt.savefig(output + ".png")
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog='Matrix generator',
        description='Generates matrices with different properties')

    parser.add_argument('-o', '--output',
                        help='Output file. Print to stdout if not specified')
    parser.add_argument('-n', '--nrows', type=int, required=True,
                        help='Number of rows in the matrix')
    parser.add_argument('-m', '--ncols', type=int,
                        help='Number of columns in the matrix. Use nrows if '
                             'not specified')
    parser.add_argument('-s', '--sparsity', type=float, default=0.7,
                        help='Sparsity level of the matrix')
    parser.add_argument('-t', '--type', default='float32',
                        choices=['float16', 'float32', 'int32', 'int64'],
                        help='Type of the values of matrix')
    parser.add_argument('-H', '--heatmap', action='store_true',
                        help='Generate the matrix heatmap')
    parser.add_argument('-p', '--pattern', default='random',
                        choices=['random', 'checkerboard', 'diagonal',
                                 'blockdiagonal', 'blockrandom', 'largerandom',
                                 'r', 'c', 'd', 'D', 'R'],
                        help='Pattern used to fill the matrix')
    parser.add_argument('-b', '--blocksize', default=32, type=int,
                        help='Blocksize of the matrix')

    args = parser.parse_args()
    if args.ncols is None:
        args.ncols = args.nrows

    if args.pattern is None or args.pattern == 'r':
        args.pattern = 'random'
    elif args.pattern == 'c':
        args.pattern = 'checkerboard'
    elif args.pattern == 'd':
        args.pattern = 'diagonal'
    elif args.pattern == 'D':
        args.pattern = 'blockdiagnonal'
    elif args.pattern == 'R':
        args.pattern = 'blockrandom'

    generate_matrix(args.output, args.nrows, args.ncols, args.sparsity,
                    args.type, args.heatmap, args.pattern, args.blocksize,)


if __name__ == "__main__":
    main()

