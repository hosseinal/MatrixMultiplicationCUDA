#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 10:00:00
##################### SLURM (do not change) ^  #####################

#!/bin/bash

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 72:00:00
#SBATCH --mem=50G  # Request 32 GB of memory

module load StdEnv/2023
module load intel/2022.2.1
echo " +-+-+-+-+ ========> ${MKLROOT}"
export MKL_DIR=$MKLROOT
echo " +-+-+-+-+ ========> ${MKL_DIR}"
module load cmake
module load gcc
module load python
module load cuda/12.2

# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# ./matrix_multiplication > ../matrix_multiplication_output.txt

./matrix_multiplication_nvbench > ../matrix_multiplication_nvbench_output.txt

# ncu --target-processes all --set full -o ./profiling_matrix_multiplication ./matrix_multiplication > results_matrix_multiplication_with_profiling.json < /dev/null
