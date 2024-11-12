#!/usr/bin/env bash

# Slurm script for running scores.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=ELLIPSE-MISTRAL
#SBATCH --output=output/%x-%A.out
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

DIRECTORY=data/
DATA=ELLIPSE_ling_feats.csv
RUBRIC=rubric.json
HF_TOKEN=/ihome/dvillarreal/alc307/ix/tokens/hugging_face.json
HF_HOME=/ihome/dvillarreal/alc307/ix/ix_models/hugging_face/

python prompt-llms.py \
    -m mistralai/Mistral-7B-Instruct-v0.2 \
    -d $DIRECTORY$DATA $DIRECTORY$RUBRIC \
    -l logs/log-noling.log debug/debug-noling.log errors/err-noling.log \
    -t $HF_TOKEN \
    -hf $HF_HOME \
