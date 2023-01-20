from corpus import Corpus
from models import TfIdfModel, LsiModel, Word2VecModel, LdaModel, LsiTfidfModel, Doc2Vec, MiniLMModel, \
    FineTunedBertModel, MultiQAMiniLMWithTorch
from preprocessing import CustomPreprocessing


class Query:

    @staticmethod
    def format_query_result(questions_df, full_df, res):
        """
        Format and print query result.
        TODO: Use Rich lib to display data in table format
        """
        for _, (doc_idx, doc_sim) in enumerate(res):
            q = questions_df.iloc[doc_idx]
            responses = full_df[full_df['ParentId'] == q['Id']]
            if len(responses.index):
                print(f'Original question: {q["Title"]} Similarity: {doc_sim}')
                print(f'Answer: {responses.iloc[0]["Body"]}\n')

    @staticmethod
    def make_query(query: str, topic: str, model: str):

        # Query preprocessing
        query = CustomPreprocessing.preprocess_query(query, tokenize=False)
        print("Query preprocessing finished")

        questions_df = Corpus.read_dataset(topic, exclude_answers=True)
        full_df = Corpus.read_dataset(topic, exclude_answers=False)

        if model == 'word2vec':
            res = Word2VecModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'tfidf':
            res = TfIdfModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lsi':
            res = LsiModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lsi-tfidf':
            res = LsiTfidfModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lda':
            res = LdaModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'doc2vec':
            res = Doc2Vec().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'mini-lm':
            res = MiniLMModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'bert':
            res = FineTunedBertModel().predict(query, topic)
            # return Query.format_query_result(questions_df, full_df, res)
        elif model == 'multi-qa-minilm-torch':
            res = MultiQAMiniLMWithTorch(topic).predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        else:
            print(f"{model} not implemented")
