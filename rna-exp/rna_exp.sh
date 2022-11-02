#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=00:20:00
#SBATCH --output=slurm_outputs/rna_exp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

#python qspright-sample-vs-nmse.py --jobid=$SLURM_JOB_ID \
#--num_subsample 2 4 6 --num_random_delays 6 8 10  --b 7 8 \
#--noise_sd 1e-4 1e-5 1e-6 1e-7 --n 25 --iters 1

#python qspright-sample-vs-nmse.py --jobid=13554482 \
#--num_subsample 2 4 6 --num_random_delays 6 8 10  --b 7 8 \
#--noise_sd 1e-4 1e-5 1e-6 1e-7 --n 25 --iters 1

python qspright-sample-vs-nmse.py --jobid=13554482 \
--num_subsample 6 --num_random_delays 10  --b 7 \
--noise_sd 1e-5 --n 25 --iters 1

#python qspright-sample-vs-nmse.py --jobid=$SLURM_JOB_ID --debug=True