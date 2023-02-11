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
    answers_df = find_answers(full_df, question_id)
    if not answers_df.empty:
        return answers_df.iloc[0]

class Query:
    @staticmethod
    def format_query_result(questions_df, full_df, res, stdout=True):
        """
        Format and print query result.
        """
        result = []
        for sort_idx, (doc_idx, doc_sim) in enumerate(res):
            question = find_question(questions_df, doc_idx)
            answer = find_first_better_answer(full_df, question['Id'])
            if answer is not None:
                result.append({
                    'idx': sort_idx + 1,
                    'question': question["Title"],
                    'best_answer': answer["Body"]
                })

        if stdout:
            print(result)

        return result

    @staticmethod
    def make_query(query: str, topic: str, model: str, generate_text=False):

        print(query, topic, model, generate_text)

        # Query preprocessing
        query_tokens = CustomPreprocessing.preprocess_query(query, tokenize=True)

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
                # TODO: create generalized function for text generation
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
        elif model == 'minilm':
            res = MiniLMModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'bert':
            res = DistilBertModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'multi-qa-minilm-torch':
            res = MultiQAMiniLMWithTorch(topic).predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        elif model == 'distilbert':
            res = DistilBertModel().predict(query, topic)
            return Query.format_query_result(questions_df, full_df, res)
        else:
            print(f"{model} not implemented")
