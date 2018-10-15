#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:59
#SBATCH --output=hmd.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment04
cd $WORK_HOME

srun -n 2 ./hmd