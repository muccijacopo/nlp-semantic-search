import os
from typing import List
from gensim import corpora, similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec, LsiModel

from corpus import Corpus

MODELS_PATH = 'models/stored_models'


def get_model_path(topic: str, model: str):
    return f'{MODELS_PATH}/{topic}_{model}_gensim.model'


def get_dictionary_path(topic: str, model: str):
    return f'{MODELS_PATH}/{topic}_{model}_gensim.dictionary'


def get_index_path(topic: str, model: str):
    return f'{MODELS_PATH}/{topic}_{model}_gensim.index'


class Models:

    @staticmethod
    def create_dictionary(corpus: List[List[str]]):
        return corpora.Dictionary(corpus)

    @staticmethod
    def corpus_to_bow(corpus: List[List[str]], dictionary: Dictionary):
        return [dictionary.doc2bow(doc) for doc in corpus]

    @staticmethod
    def doc_to_bow(doc: List[List[str]], dictionary: Dictionary):
        return dictionary.doc2bow(doc)

    @staticmethod
    def compute_word2vec_similarity(model: Word2Vec, ws1: List[str], ws2: List[str]):
        """Compute cosine similarity between two sets of words."""
        return model.wv.n_similarity(ws1, ws2)

    @staticmethod
    def train_gensim_tfidf(topic: str):
        corpus = Corpus.get_corpus(topic)
        dictionary = Models.create_dictionary(corpus)
        # convert corpus to BoW format
        bow_corpus = Models.corpus_to_bow(corpus, dictionary)

        # fit model
        tfidf = TfidfModel(bow_corpus)
        nf = len(dictionary.dfs)
        index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=nf)

        tfidf.save(f'{MODELS_PATH}/{topic}_tfidf_gensim.model')
        index.save(f'{MODELS_PATH}/{topic}_tfidf_gensim.index')
        dictionary.save(f'{MODELS_PATH}/{topic}_tfidf_gensim.dictionary')

    @staticmethod
    def predict_gensim_tfidf(query: str, topic: str):

        # load model, index and dictionary
        tfidf: TfidfModel = TfidfModel.load(f'{MODELS_PATH}/{topic}_tfidf_gensim.model')
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(f'{MODELS_PATH}/{topic}_tfidf_gensim.index')
        dictionary: corpora.Dictionary = corpora.Dictionary.load(f'{MODELS_PATH}/{topic}_tfidf_gensim.dictionary')

        # convert query to bow format
        query_bow = Models.doc_to_bow(query, dictionary)

        # Compute similarity between query and this index
        sims = index[tfidf[query_bow]]
        # Similarity between query and each document sorted

        # Return first 10 most similar documents
        return [(doc_idx, doc_sim) for doc_idx, doc_sim in
                sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:10]]

    @staticmethod
    def word2vec(query: List[str], corpus: List[List[str]]):

        # Training algorithm: 1 for skip-gram; otherwise CBOW.
        word2vec = Word2Vec(corpus, min_count=10, sg=0, window=10)

        # Precompute L2-normalized vectors.
        # If replace is set, forget the original vectors and only keep the normalized ones = saves lots of memory!
        # Note that you cannot continue training after doing a replace. The model becomes effectively read-only = you can call most_similar, similarity etc., but not train.
        word2vec.init_sims(replace=True)

        # Return first 10 most similar documents
        return sorted([(doc_idx, word2vec.wv.n_similarity(query, doc_content)) for doc_idx, doc_content in enumerate(corpus) if len(doc_content) != 0], key=lambda x: x[1], reverse=True)[:10]


    @staticmethod
    def train_gensim_lsi(topic: str):
        corpus = Corpus.get_corpus(topic)
        dictionary = Models.create_dictionary(corpus)
        bow_corpus = Models.corpus_to_bow(corpus, dictionary)
        lsi = LsiModel(bow_corpus, id2word=dictionary, num_topics=10)
        # transform corpus to LSI space and index it
        index = similarities.MatrixSimilarity(lsi[bow_corpus])

        lsi.save(get_model_path(topic, 'lsi'))
        index.save(get_index_path(topic, 'lsi'))
        dictionary.save(get_dictionary_path(topic, 'lsi'))

    @staticmethod
    def predict_gensim_lsi(query: List[str], topic: str):
        print(topic)
        lsi = LsiModel.load(get_model_path(topic, 'lsi'))
        dictionary = corpora.Dictionary.load(get_dictionary_path(topic, 'lsi'))
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(get_index_path(topic, 'lsi'))

        query_bow = Models.doc_to_bow(query, dictionary)
        query_lsi = lsi[query_bow]

        # perform a similarity query against the corpus
        sims = index[query_lsi]

        r = [(doc_idx, doc_sim) for doc_idx, doc_sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
        return r[:10]
