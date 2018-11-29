#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2 
#SBATCH --gres=gpu:2 
#SBATCH --time=00:00:59 
#SBATCH --output=pi3.out 
#SBATCH -A lc_an2

WORK_HOME=/auto/rcf-40/liangsiq/assignment06
cd $WORK_HOME
srun –n 2 ./pi3
