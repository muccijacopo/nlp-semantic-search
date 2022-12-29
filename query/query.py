from corpus import Corpus
from models import Models
from preprocessing import CustomPreprocessing


class Query:

    @staticmethod
    def format_query_result(questions_df, full_df, res):
        """
        Format and print query result.
        TODO: Use Rich lib to display data in table format
        """
        for index, (doc_idx, doc_sim) in enumerate(res):
            q = questions_df.iloc[doc_idx]
            responses = full_df[full_df['ParentId'] == q['Id']]
            # print(f"{index+1}) {questions_df.iloc[doc_idx]['Title']} | Similarity: {doc_sim}")
            if len(responses.index):
                print('Origial question: ', q['Title'], '\n')
                print(responses.iloc[0]['Body'])
                print('\n\n\n')

    @staticmethod
    def make_query(query: str, topic: str, model: str):

        print(f"Query: {query}")
        print(f"Topic: {topic}")

        # Query preprocessing
        query = CustomPreprocessing.preprocess_query(query)
        print("Query preprocessing finished")

        questions_df = Corpus.read_dataset(topic, exclude_answers=True)
        full_df = Corpus.read_dataset(topic, exclude_answers=False)

        if model == 'word2vec':
            # res = Models.word2vec(query, corpus)
            # return Query.format_query_result(df, res)
            print(model)
        elif model == 'tfidf':
            res = Models.predict_gensim_tfidf(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lsi' or model == 'lsa':
            res = Models.predict_gensim_lsi(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        else:
            print(f"{model} not implemented")
