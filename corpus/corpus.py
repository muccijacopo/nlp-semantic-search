import pandas as pd

from preprocessing import CustomPreprocessing


class Corpus:
    @staticmethod
    def get_corpus(topic: str):
        # Read dataset
        df = Corpus.read_dataset(topic)

        df['Title__Preprocessed'] = CustomPreprocessing.reprocess_title(df['Title'].copy())
        corpus = CustomPreprocessing.tokenize_list_of_sentences(df['Title__Preprocessed'].values)
        return corpus

    @staticmethod
    def read_dataset(topic: str):
        """Read dataset from csv file."""
        try:
            df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')
            return df
        except FileNotFoundError:
            raise Exception('Topic not found')
        except Exception:
            raise Exception("Unknown error during dataset read")
