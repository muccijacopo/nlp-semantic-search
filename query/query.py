from corpus import Corpus
from models import Models
from preprocessing import CustomPreprocessing


class Query:

    @staticmethod
    def format_query_result(df, res):
        """
        Format and print query result.
        TODO: Use Rich lib to display data in table format
        """
        print(res)
        for index, (doc_idx, doc_sim) in enumerate(res):
            print(f"{index+1}) {df.iloc[doc_idx]['Title']} | Similarity: {doc_sim}")


    @staticmethod
    def make_query(query: str, topic: str, model: str):

        print(f"Query: {query}")
        print(f"Topic: {topic}")

        # Query preprocessing
        query = CustomPreprocessing.preprocess_query(query)
        print("Query preprocessing finished")

        df = Corpus.read_dataset(topic)

        if model == 'word2vec':
            # res = Models.word2vec(query, corpus)
            # return Query.format_query_result(df, res)
            print(model)
        elif model == 'tfidf':
            res = Models.predict_gensim_tfidf(query, topic)
            return Query.format_query_result(df, res)
        elif model == 'lsi' or model == 'lsa':
            res = Models.predict_gensim_lsi(query, topic)
            return Query.format_query_result(df, res)
        else:
            print(f"{model} not implemented")
