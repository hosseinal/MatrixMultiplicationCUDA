# !/usr/bin/python3

import argparse
import numpy as np
from scipy import sparse
from scipy.io import mmwrite
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
                    blocksize: int = 16, block_rows: int = None, block_cols: int = None):

    matrix = np.zeros((num_rows, num_cols), dtype=dtype)

    def fill_block(block_i, block_j):
        # choose block sizes: priority block_rows/block_cols if provided,
        # otherwise square blocksize is used for both dimensions
        br = block_rows if block_rows is not None else blocksize
        bc = block_cols if block_cols is not None else blocksize
        # fill the matrix from i to i + br and j to j + bc
        start_row = block_i * br
        start_col = block_j * bc
        end_row = min(start_row + br, num_rows)
        end_col = min(start_col + bc, num_cols)

        block_shape = (end_row - start_row, end_col - start_col)

        matrix[start_row:end_row, start_col:end_col] = generate_dense_matrix(
            block_shape, dtype)

    if pattern == 'random':
        matrix = sparse.random(num_rows, num_cols, (1.0 - sparsity),
                               dtype=dtype if dtype != 'float16' else 'float32'
                               ).toarray().astype(dtype)
    elif pattern == 'checkerboard':
        br = block_rows if block_rows is not None else blocksize
        bc = block_cols if block_cols is not None else blocksize
        num_block_rows = (num_rows + br - 1) // br
        num_block_cols = (num_cols + bc - 1) // bc
        for i in range(num_block_rows):
            for j in range(num_block_cols):
                if (i % 2) ^ (j % 2) == 0:
                    fill_block(i, j)

    elif pattern == 'diagonal':
        for i in range(min(num_rows, num_cols)):
            matrix[i:i+1, i:i+1] = generate_dense_matrix((1,1), dtype)

    elif pattern == 'blockdiagonal':
        br = block_rows if block_rows is not None else blocksize
        bc = block_cols if block_cols is not None else blocksize
        num_block_diag = min(num_rows, num_cols) // max(br, bc) + 1
        for i in range(num_block_diag):
            fill_block(i, i)

    elif pattern == 'blockrandom':
        br = block_rows if block_rows is not None else blocksize
        bc = block_cols if block_cols is not None else blocksize
        num_block_rows = (num_rows + br - 1) // br
        num_block_cols = (num_cols + bc - 1) // bc
        mask = sparse.random(num_block_rows,
                             num_block_cols,
                             (1.0 - sparsity), dtype='int32').toarray()
        for i in range(num_block_rows):
            for j in range(num_block_cols):
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

    # If the output filename ends with .mtx, write as Matrix Market sparse file.
    if output is not None and output.lower().endswith('.mtx'):
        # Convert to a sparse COO matrix to store only nonzeros
        # Ensure numeric dtype is supported (promote float16 to float32)
        if hasattr(matrix, 'dtype') and str(matrix.dtype) == 'float16':
            mat_to_write = matrix.astype('float32')
        else:
            mat_to_write = matrix
        mat_sparse = sparse.coo_matrix(mat_to_write)
        mmwrite(output, mat_sparse)
    else:
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
    parser.add_argument('-br', '--blockrows', default=None, type=int,
                        help='Block row size (height) of the matrix')
    parser.add_argument('-bc', '--blockcols', default=None, type=int,
                        help='Block column size (width) of the matrix')

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
        args.pattern = 'blockdiagonal'
    elif args.pattern == 'R':
        args.pattern = 'blockrandom'

    generate_matrix(args.output, args.nrows, args.ncols, args.sparsity,
                    args.type, args.heatmap, args.pattern,
                    args.blocksize, args.blockrows, args.blockcols)


if __name__ == "__main__":
    main()

