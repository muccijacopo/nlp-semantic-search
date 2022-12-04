import pandas as pd

from models import Models
from preprocessing import Preprocessing


class Query:
    @staticmethod
    def make_query(query: str, topic: str = 'beer', model: str = 'word2vec'):
        # Read dataset
        try:
            df = pd.read_csv(f'./data/stackechange_csv/{topic}.stackexchange.com-posts.csv', sep=',')
        except FileNotFoundError as e:
            return print('Topic not found')
        except Exception:
            return print("Unknown error during dataset read")

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

        if model == 'word2vec':
            return Models.word2vec(query, corpus)
        elif model == 'tfidf':
            return Models.tfidf(query, corpus)
        elif model == 'lsi' or model == 'lsa':
            return Models.latent_semantic_indexing(query, corpus)
        else:
            print(f"{model} not implemented")
