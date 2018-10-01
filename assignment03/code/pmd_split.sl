#!/bin/bash
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --time=00:04:59
#SBATCH --output=pmd_split.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment03
cd $WORK_HOME

mpirun -np $SLURM_NTASKS ./pmd_split
