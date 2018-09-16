#!/bin/bash
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:59
#SBATCH --out=global_pi.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment02
cd $WORK_HOME

mpirun -n $SLURM_NTASKS ./global_pi
mpirun -n             4 ./global_pi
mpirun -n             2 ./global_pi
mpirun -n             1 ./global_pi
