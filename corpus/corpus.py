import pandas as pd

from preprocessing import CustomPreprocessing
from utils import memory_manager


class Corpus:

    @staticmethod
    def get_corpus(topic: str):
        # Get corpus from dataset file
        df = Corpus.read_dataset(topic, exclude_answers=True)
        df['Title__Preprocessed'] = CustomPreprocessing.preprocess_content(df['Title'].copy())
        df['Body__Reprocessed'] = CustomPreprocessing.preprocess_content(df['Body'].copy())
        df['QuestionContent'] = df['Title__Preprocessed'] + ' ' + df['Body__Reprocessed']
        corpus = CustomPreprocessing.tokenize_list_of_sentences(df['QuestionContent'].values)
        return corpus

    @staticmethod
    def read_dataset(topic: str, exclude_answers):
        """Read dataset from csv file."""
        try:
            df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')
            if exclude_answers:
                df = df[df['Title'].notna()]
            return df
        except FileNotFoundError:
            raise Exception('Topic not found')
        except Exception:
            raise Exception("Unknown error during dataset read")
