# Prompting the Mistrel LLM on the Feedback Prize Datasets
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path
from prompts import (make_prompt,
                     batch_prompts,
                     make_rubric, )
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from vllm import (LLM,
                  SamplingParams, )

import argparse
import json
import logger
import logging
import os

import pandas as pd


log, debug, err = logging.getLogger(), logging.getLogger(), logging.getLogger()


def set_environment(token: str, model_store: str):

    os.environ['HF_TOKEN'] = json.load(token)['token']
    debug.debug(f"HF_TOKEN set to {token}")

    if model_store != "":

        os.environ['HF_HOME'] = model_store
        debug.debug(f"Set HF_HOME to {model_store}")


def main(args: argparse.Namespace):

    set_environment(args.token, args.huggingface)

    test_df = pd.read_csv(args.data[0])
    rubric = make_rubric(args.data[1])

    # Generate an example prompt
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

    prompts = list(
        batch_prompts(
            rubric=rubric, 
            scoring_range=(1, 5),
            essay_prompts=test_df['prompt'],
            essays=test_df['full_text'],
            additional_information = "",
            model_prefix="", 
            model_suffix="",
            )
        )

    outputs = llm.generate(prompts, sampling_params)

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

    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Path to the JSON file containing the HuggingFace access token under 'token'.\n \n",
    )

    parser.add_argument(
        "-hf",
        "--huggingface",
        type=str,
        default="",
        help="Path where the model should be stored if it is a HuggingFace model.\n \n",
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

    log, debug, err = logger.make_loggers(
        *args.logging, 
        levels=[logging.INFO,
                logging.DEBUG,
                logging.WARNING],
        )
    
    print(log, debug, err)

    main(args)
