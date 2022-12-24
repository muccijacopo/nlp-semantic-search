import pandas as pd

from models import Models
from preprocessing import CustomPreprocessing


class Query:

    @staticmethod
    def format_query_result(df, res):
        """
        Format and print query result.
        TODO: Use Rich lib to display data in table format
        """
        [(df['Title'].values[doc_idx], doc_sim) for doc_idx, doc_sim in res]
        for index, (doc_idx, doc_sim) in enumerate(res):
            print(f"{index+1}) {df['Title'].values[doc_idx][:100]} Similarity: {doc_sim}")

    @staticmethod
    def make_query(query: str, topic: str, model: str):

        print(f"Query: {query}")
        print(f"Topic: {topic }")

        # Read dataset
        try:
            df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')
        except FileNotFoundError as e:
            return print('Topic not found')
        except Exception:
            return print("Unknown error during dataset read")

        # Dataset preprocessing
        # TODO: try different type of Reprocessing (ex. simple_reprocess from Gensim)
        df['Title__Preprocessed'] = CustomPreprocessing.reprocess_title(df['Title'].copy())
        print("Dataset preprocessing finished")

        # Corpus creation
        # TODO: try to decrease tokenization step execution time
        corpus = CustomPreprocessing.tokenize_list_of_sentences(df['Title__Preprocessed'].values)
        print("Dataset tokenization finished. Corpus initialized")

        # Query preprocessing
        query = CustomPreprocessing.preprocess_query(query)
        print("Query preprocessing finished")

        if model == 'word2vec':
            res = Models.word2vec(query, corpus)
            return Query.format_query_result(df, res)
        elif model == 'tfidf':
            res = Models.tfidf(query, corpus)
            return Query.format_query_result(df, res)
        elif model == 'lsi' or model == 'lsa':
            res = Models.latent_semantic_indexing(query, corpus)
            return Query.format_query_result(df, res)
        else:
            print(f"{model} not implemented")
