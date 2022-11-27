import pandas as pd

from embeddings import Embeddings
from preprocessing import Preprocessing


class Query:
    @staticmethod
    def make_query(query: str):
        df = pd.read_csv('./data/stackechange_csv/datascience.stackexchange.com-posts.csv', sep=',')
        df['Title'] = df['Title'].apply(Preprocessing.apply_lowercase)
        df['Title'] = df['Title'].apply(Preprocessing.remove_special_characters)
        df['Title'] = df['Title'].apply(Preprocessing.remove_html)

        # Tokenization
        # TODO: move Tokenization step in Preprocessing class
        corpus = []
        for title in df['Title'].values:
            words = Preprocessing.tokenize(title)
            corpus.append(words)
        corpus = [Preprocessing.remove_stopwords(words) for words in corpus]

        return Embeddings.word2vec_similarity(query, corpus)
