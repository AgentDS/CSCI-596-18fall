#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:59
#SBATCH --output=pdf1.out
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment06
cd $WORK_HOME
srun –n 1 ./pdf1
