#!/bin/bash

#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 00:01:00
#SBATCH --mem=50G  # Request 32 GB of memory


module load StdEnv/2023
module load intel/2022.2.1
echo " +-+-+-+-+ ========> ${MKLROOT}"
export MKL_DIR=$MKLROOT
echo " +-+-+-+-+ ========> ${MKL_DIR}"
module load cmake
module load gcc
module load cuda/12.2


mkdir build
cd build
cmake ..
make

./matrix_multiplication > result.txt