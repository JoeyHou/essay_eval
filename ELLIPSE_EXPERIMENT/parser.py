# Parse the model output for scores
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path

import argparse
import logger
import logging
import re

import pandas as pd


log = logging.getLogger()


def main(args: argparse.Namespace):
    
    with open(args.data, 'r') as src:
        lines = [line.strip() for line in src]

    SCORES = re.compile(r'#+ ((?:Cohesion|Syntax|Vocabulary|Phraseology|Grammar|Conventions|Overall): \d)', re.I)
    CATS = ["COHESION", "SYNTAX", "VOCABULARY", "PHRASEOLOGY", "GRAMMAR", "CONVENTIONS", "OVERALL"]

    records = []
    for line in lines:

        score_key = {text.split(":")[0].upper(): int(text.split(":")[1]) for text in SCORES.findall(line)}
        records.append({cat : score_key[cat] if cat in score_key else pd.NA for cat in CATS})

    df = pd.DataFrame.from_records(records)

    print(df.info())


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Data paths leading to the log files with the generated text.\n \n",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        required=True,
        help="CSV file to store the parsed outputs.\n \n",
    )

    parser.add_argument(
        "-l",
        "--logging",
        type=Path,
        required=True,
        help="Paths to the main logger.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="prompts-llms.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run prompt-based LLMs via vllm.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    log, = logger.make_loggers(
        args.logging, 
        levels=logging.INFO,
        )
    
    print(log)

    main(args)
