#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:59
#SBATCH --output=hmd_batch.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment04
cd $WORK_HOME

mpicc -O -o hmd1thrd hmd1.c -lm -fopenmp
srun -n 1 ./hmd1thrd

mpicc -O -o hmd2thrd hmd2.c -lm -fopenmp
srun -n 1 ./hmd2thrd

mpicc -O -o hmd4thrd hmd3.c -lm -fopenmp
srun -n 1 ./hmd4thrd

mpicc -O -o hmd8thrd hmd4.c -lm -fopenmp
srun -n 1 ./hmd8thrd
