#! usr/bin/bash

DIRECTORY=data/ELLIPSE/
DATA=ELLIPSE_Final_github.csv
RUBRIC=rubric.json

python prompt-llms.py \
    -m mistralai/Mistral-7B-Instruct-v0.2 \
    -d $DIRECTORY$DATA $DIRECTORY$RUBRIC \
    -l logs/logs1.log debug/debug1.log errors/errs1.log
