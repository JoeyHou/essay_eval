# Calculate the necessary linguistic features
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path

import argparse
import features

import pandas as pd


def main(args: argparse.Namespace):

    df = pd.read_csv(args.data)

    feats = features.FeatureSet(df=df, essay_col=args.text)
    feats.all_feats([
        'complex_words_dc',
        'characters',
        'long_words',
        ])
    
    df = feats.df.drop(columns=["TOKENS"])
    df.to_csv(args.save)

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Data paths leading to the CSV data.\n \n",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="text",
        help="Column in the data which contains the full essay text.\n \n",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=str,
        required=True,
        help="Save name and location of the new linguistic features.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=f"{__file__.split('/')[-1]}.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extract linguistic features from an essay corpus.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
