import json

from corpus import Corpus
from models import TfIdfModel, LsiModel, Word2VecModel, LdaModel, LsiTfidfModel, Doc2Vec, MiniLMModel, \
    FineTunedBertModel, MultiQAMiniLMWithTorch, DistilBertModel, QuestionAnsweringDistilbertModel
from preprocessing import CustomPreprocessing


def find_question(questions_df, doc_idx):
    q = questions_df.iloc[doc_idx]
    return q


def find_answers(full_df, question_id):
    responses = full_df[full_df['ParentId'] == question_id]
    return responses


def find_first_better_answer(full_df, question_id):
    answers = find_answers(full_df, question_id)
    return answers.iloc[0]


class Query:
    @staticmethod
    def format_query_result(questions_df, full_df, res, stdout=True):
        """
        Format and print query result.
        TODO: Use Rich lib to display data in table format
        TODO: Add support for HTML
        """
        s = ""
        for sort_idx, (doc_idx, doc_sim) in enumerate(res):
            question = find_question(questions_df, doc_idx)
            answer = find_first_better_answer(full_df, question['Id'])
            s += f'{sort_idx+1}) Original question: {question["Title"]} Similarity: {doc_sim} \n'
            s += f'Answer: {answer["Body"]}\n\n'

        if stdout:
            print(s)
        return s

    @staticmethod
    def make_query(query: str, topic: str, model: str, generate_text=False):

        # Query preprocessing
        query_tokens = CustomPreprocessing.preprocess_query(query, tokenize=True)
        print("Query preprocessing finished")

        questions_df = Corpus.read_dataset(topic, exclude_answers=True)
        full_df = Corpus.read_dataset(topic, exclude_answers=False)

        if model == 'word2vec':
            res = Word2VecModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'tfidf':
            res = TfIdfModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lsi':
            res = LsiModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lsi-tfidf':
            res = LsiTfidfModel().predict(query_tokens, topic)
            if generate_text:
                first_doc_idx, _ = res[0]
                question = find_question(questions_df, first_doc_idx)
                answers = find_answers(full_df, question['Id'])
                context = ""
                for _, answer_row in answers.iterrows():
                    context += f"{CustomPreprocessing.simple_preprocess(answer_row['Body'], lowercase=False, tokenize=False, special_characters=True)} \n"
                res = QuestionAnsweringDistilbertModel().generate(query, context)
                print(res)
                return json.dumps(res)
            else:
                return Query.format_query_result(questions_df, full_df, res)
        elif model == 'lda':
            res = LdaModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'doc2vec':
            res = Doc2Vec().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'mini-lm':
            res = MiniLMModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'bert':
            pass
            # res = FineTunedBertModel().predict(query, topic)
            # return Query.format_query_result(questions_df, full_df, res)
        elif model == 'multi-qa-minilm-torch':
            res = MultiQAMiniLMWithTorch(topic).predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'distilbert':
            res = DistilBertModel().predict(query_tokens, topic)
            return Query.format_query_result(questions_df, full_df, res)
        else:
            print(f"{model} not implemented")
