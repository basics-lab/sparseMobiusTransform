#!/bin/bash
#SBATCH --job-name=synt_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

python run-tests-nmse-vs-snr.py --jobid=$SLURM_JOB_ID --debug False \
--num_subsample 3 --num_repeat 1 --b 10 --t 6 8 10 12 --n 500 --sparsity 500 --iters 1
