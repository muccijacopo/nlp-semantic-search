import pandas as pd

from preprocessing import CustomPreprocessing, GensimPreprocessing
from utils import memory_manager


class Corpus:

    @staticmethod
    def get_corpus(topic: str, title_only=False, tokenize=True):
        # Get corpus from dataset file
        df = Corpus.read_dataset(topic, exclude_answers=True)
        if title_only:
            df['QuestionContent'] = df['Title']
        else:
            df['QuestionContent'] = df['Title'] + ' ' + df['Body']
        corpus = [CustomPreprocessing.simple_preprocess(d, tokenize) for d in df['QuestionContent'].values]
        # corpus = [GensimPreprocessing.simple_preprocess(d) for d in df['QuestionContent'].values]
        return corpus

    @staticmethod
    def read_dataset(topic: str, exclude_answers):
        """Read dataset from csv file."""
        try:
            df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')
            print(f"Reading {topic} dataset. Number of records: {len(df.index)}")
            if exclude_answers:
                df = df[df['Title'].notna()]
            return df
        except FileNotFoundError:
            raise Exception('Topic not found')
        except Exception:
            raise Exception("Unknown error during dataset read")
