#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --time=00:00:59
#SBATCH --output=global.out
#SBATCH -A lc_an2
WORK_HOME=/home/rcf-proj/an2/Your_ID
cd $WORK_HOME
srun -n $SLURM_NTASKS --mpi=pmi2 ./global
srun -n             4 --mpi=pmi2 ./global
