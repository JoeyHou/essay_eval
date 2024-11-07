#!/usr/bin/env bash

# Slurm script for running scores.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=ELLIPSE-MISTRAL
#SBATCH --output=output/%x.out
#SBATCH --mail-user=alc307@pitt.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --qos=short
############## SBATCH HEADER END ##############

echo "RUN:" `date`

# Load necessary modules
module load gcc/8.2.0 python/anaconda3.10-2022.10

# Activate the conda environment
source activate mistral

unset PYTHONHOME
unset PYTHONPATH

echo "RUN: `date`"

DIRECTORY=data/ELLIPSE/
DATA=ELLIPSE_Final_github.csv
RUBRIC=rubric.json

python prompt-llms.py \
    -m mistralai/Mistral-7B-Instruct-v0.2 \
    -d $DIRECTORY$DATA $DIRECTORY$RUBRIC \
    -l logs/log1.log debug/debug1.log errors/err1.log
