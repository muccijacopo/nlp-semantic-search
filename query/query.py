import pandas as pd

from embeddings import Embeddings
from preprocessing import Preprocessing


class Query:
    @staticmethod
    def make_query(query: str, topic: str = 'beer'):
        # Read dataset
        df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')

        # TODO: move all preprocessing steps in Preprocessing class (rename to CustomReprocessing)
        # TODO: try different type of Reprocessing (ex. simple_reprocess from Gensim)
        # Dataset preprocessing
        df['Title'] = df['Title'].apply(Preprocessing.apply_lowercase)
        df['Title'] = df['Title'].apply(Preprocessing.remove_special_characters)
        df['Title'] = df['Title'].apply(Preprocessing.remove_html)

        print("Dataset preprocessing finished")

        # TODO: try to decrease tokenization step execution time
        # Create corpus from dataset (tokenization)
        corpus = Preprocessing.tokenize_list_of_sentences(df['Title'].values)
        print("Dataset tokenization finished. Corpus initialized")

        # Query preprocessing
        query = Preprocessing.apply_lowercase(query)
        query = Preprocessing.remove_special_characters(query)
        query = Preprocessing.remove_stopwords(Preprocessing.tokenize_sentence(query))
        print("Query preprocessing finished")

        return Embeddings.word2vec_similarity(query, corpus)
