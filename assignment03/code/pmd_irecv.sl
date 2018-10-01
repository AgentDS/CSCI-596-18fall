#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4
#SBATCH --time=00:01:59
#SBATCH --output=pmd_irecv.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment03
cd $WORK_HOME

counter=0
while [ $counter -lt 3 ]; do
  echo "***** Asynchronous *****"
  mpirun -np $SLURM_NTASKS ./pmd_irecv
  echo "***** Synchronous *****"
  mpirun -np $SLURM_NTASKS ./pmd
  let counter+=1
done
