# Prompting the Mistrel LLM on the Feedback Prize Datasets
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path
from prompts import (make_prompt,
                     make_rubric, )
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from vllm import (LLM,
                  SamplingParams, )

import argparse
import logger
import logging

import pandas as pd


def main(args: argparse.Namespace):

    log, debug, err = logger.make_loggers(
        *args.logging, 
        levels=[logging.INFO,
                logging.DEBUG,
                logging.WARNING],
        )
    
    print(log, debug, err)

    test_df = pd.read_csv(args.data[0])
    rubric = make_rubric(args.data[1])

    prompt = make_prompt(
        rubric=rubric, 
        scoring_range=(1, 5),
        essay_prompt=test_df['prompt'][0],
        essay=test_df['full_text'][0],
        model_prefix="", 
        model_suffix="",
        )
    
    debug.debug(f"=====================PROMPT EXAMPLE=====================\n{prompt.format()}")

    llm = LLM(model=args.models[0])
    sampling_params = SamplingParams(temperature=0.01, max_tokens=4096)  # As in Joey's eval.py

    outputs = llm.generate(prompt.format(), sampling_params)

    # Print the outputs.
    for output in tqdm(outputs, desc="Running model on dataset..."):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        log.info(f"Generated text: {generated_text!r}")


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        nargs=2,
        required=True,
        help="Data paths leading to the CSV data and then the JSON rubric.\n \n",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        default=["facebook/opt-125m"],  # Will not treat it as a 1-element array otherwise
        help="Data required.\n \n",
    )

    parser.add_argument(
        "-l",
        "--logging",
        type=Path,
        nargs=3,
        required=True,
        help="Paths to the main logger, debug logger and error logger.\n \n",
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

    main(args)
