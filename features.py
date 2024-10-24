# Class to store and calculate features
# Created by Alejandro Ciuba, alc307@pitt.edu
# Code based on https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/main/features.py
from collections import Counter
from tqdm import tqdm

import nltk
import readability
import spacy
import string
# import syllables

import pandas as pd

NLP: spacy.Language = spacy.load("en_core_web_sm")

# NO LOWERCASE NORMALIZATION PERFORMED
class FeatureSet:

    df: pd.DataFrame
    essay_col: str
    token_col: str

    def __init__(self, df: pd.DataFrame, essay_col: str = "text", token_col: str = "TOKENS") -> None:

        self.df = df
        self.essay_col = essay_col
        self.token_col = token_col
        self.df[self.token_col] = self.df[self.essay_col].map(lambda x: nltk.word_tokenize(self.normalize_text(x)))  # No punctuation

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Removes punctuation as in https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/main/features.py#L29
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def hapax(self, col_name: str = "HAPAX"):
        """
        Based on https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/main/features.py#L207 `unique_words`.

        Issues
        ---
        - No casing normalization.

        Returns
        ---

        The number of hapax legomena in all essays under `HAPAX` or `col_name`. 
        """
        
        # TODO: Needs to be tested
        def hapax(bow: list[str]) -> int:
            return len(list(filter(lambda x: True if col[x] == 1 else False, col := Counter(bow))))
        
        self.df[col_name] = self.df[self.token_col].map(hapax)

    def ess_char_len(self, col_name: str = "ESS_CHAR_LEN"):
        """
        Based on https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/main/features.py#L133 `char_count`

        Returns
        ---
        
        The number of non-space, non-punctuation characters in all essays under `HAPAX` or `col_name`.
        """
        self.df[col_name] = self.df[self.token_col].map(lambda x: Counter(x).total())

    def word_count(self, col_name: str = "WORD_COUNT"):
        """
        Returns
        ---

        The number of words in all essays under `WORD_COUNT` or `col_name`.
        """
        self.df[col_name] = self.df[self.token_col].map(len)

    def sentence_count(self, col_name: str = "SENTENCE_COUNT"):
        """
        Returns
        ---

        The number of sentences in all essays under `SENTENCE_COUNT` or `col_name`.
        """
        self.df[col_name] = self.df[self.essay_col].map(lambda x: len(nltk.tokenize.sent_tokenize(x)))

    # TODO: syllables does one word at a time, so I want to find a bulkier library
    def syllable_count(self, col_name: str = "SYLLABLE_COUNT"):
        """
        Returns
        ---

        The number of syllables in all essays under `SYLLABLE_COUNT` or `col_name`.
        """
        pass

    # Everything is unique (set(...))
    def spacy_measures(self):
        """
        Returns
        ---

        All measures gotten from utilizing spacy's `en_core_web_sm`.
        """

        # TODO: Test
        def get_measures(text: str) -> list[str]:

            doc = NLP(text)

            lemma_count = len(set([token.lemma_ for token in doc]))
            noun_count = len(set([token.text for token in doc if token.pos_ in ["PROPN", "NOUN"]]))
            stop_word_count = len(set([token.text for token in doc if token.is_stop]))

            return [lemma_count, noun_count, stop_word_count]
        
        columns = ["LEMMA_COUNT", "NOUN_COUNT", "STOP_WORD_COUNT"]
        self.df[columns] = pd.DataFrame(self.df[self.essay_col].map(lambda x: get_measures(x)).to_list())


    def readability_measures(self, 
                             feats: dict[str, str] | list[str] = ['complex_words_dc',
                                                                  'characters',
                                                                  'long_words',
                                                                  ]):
        """
        Returns
        ---
        The readability features provided either in the value column of a dictionary or the `str.upper()` version of the name.
        """

        def get_measures(text: str, measures: list) -> list[int]:

            values = readability.getmeasures(text)['sentence info']
            return [values[measure] for measure in measures]

        # Takes a Series containing a list and puts each element i into its own column i
        # https://datagy.io/pandas-split-column-of-lists-into-columns/
        if isinstance(feats, dict):
            self.df[list(feats.values())] = pd.DataFrame(self.df[self.essay_col].map(lambda x: get_measures(x, measures=list(feats.keys()))).to_list())
        elif isinstance(feats, list):
            self.df[list(map(str.upper, feats))] = pd.DataFrame(self.df[self.essay_col].map(lambda x: get_measures(x, measures=feats)).to_list())

    def all_feats(self, *args):
        """
        Calculate all features.
        """
    
        funcs = [
            self.hapax,
            self.word_count,
            self.sentence_count,
            self.ess_char_len,
            self.spacy_measures,
            self.readability_measures,
        ]

        for i, func in enumerate(tqdm(func, desc="Running all feature generation methods...")):

            if i == len(funcs) - 1:
                func(args)
            else:
                func()


if __name__ == "__main__":

    df = pd.read_csv('INSERT PATH HERE')
    df = df.loc[0:100]

    feats = FeatureSet(df=df, essay_col='INSERT COLUMN HERE')

    feats.hapax()
    feats.word_count()
    feats.sentence_count()
    feats.ess_char_len()
    feats.spacy_measures()
    feats.readability_measures([
        'complex_words_dc',
        'characters',
        'long_words',
    ])

    print(feats.df.loc[0, :])
