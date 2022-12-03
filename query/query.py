import pandas as pd

from embeddings import Embeddings
from preprocessing import Preprocessing


class Query:
    @staticmethod
    def make_query(query: str):
        # Read dataset
        df = pd.read_csv('./data/stackechange_csv/beer.stackexchange.com-posts.csv', sep=',')

        # Preprocessing
        df['Title'] = df['Title'].apply(Preprocessing.apply_lowercase)
        df['Title'] = df['Title'].apply(Preprocessing.remove_special_characters)
        df['Title'] = df['Title'].apply(Preprocessing.remove_html)

        # Create corpus from dataset (tokenization)
        corpus = Preprocessing.tokenize_list_of_sentences(df['Title'].values)

        # Query preprocessing
        query = Preprocessing.apply_lowercase(query)
        query = Preprocessing.remove_special_characters(query)
        query = Preprocessing.remove_stopwords(Preprocessing.tokenize_sentence(query))

        return Embeddings.word2vec_similarity(query, corpus)
