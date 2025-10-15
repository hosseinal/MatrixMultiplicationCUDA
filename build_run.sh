#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 01:00:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


# build the benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

./matrix_multiplication > ../matrix_multiplication_output.txt

ncu --target-processes all --set full --benchmark_report_aggregates_only=true --benchmark_format=json -o ./profiling_matrix_multiplication ./matrix_multiplication > results_matrix_multiplication_with_profiling.json < /dev/null


