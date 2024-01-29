#!/bin/bash
#SBATCH --job-name=synt_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=72:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=landonb@berkeley.edu

python run-tests-complexity-vs-size.py --jobid=$SLURM_JOB_ID \
--num_subsample 3 --num_repeat 1 --b 1 2 3 4 5 6 7 8 9 \
--a 1 --snr 0 --n 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 --q 2 --sparsity 100 --iters 5

#python run-tests-nmse-vs-snr.py --jobid=$SLURM_JOB_ID \
#--num_subsample 3 --num_repeat 1 --b 7 \
#--a 1 --n 20 --q 4 --sparsity 100 250 1000 --iters 5
