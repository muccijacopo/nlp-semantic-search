from typing import List
from abc import abstractmethod
from gensim import corpora, similarities, models
from gensim.corpora import Dictionary

from corpus import Corpus


class Model:
    MODELS_PATH = 'models/stored_models'

    @abstractmethod
    def train(self, topic: str):
        pass

    @abstractmethod
    def predict(self, query: str, topic: str):
        pass

    def get_model_path(self, topic: str, model: str):
        return f'{self.MODELS_PATH}/{topic}_{model}_gensim.model'

    def get_dictionary_path(self, topic: str, model: str):
        return f'{self.MODELS_PATH}/{topic}_{model}_gensim.dictionary'

    def get_index_path(self, topic: str, model: str):
        return f'{self.MODELS_PATH}/{topic}_{model}_gensim.index'

    def create_dictionary(self, corpus: List[List[str]]):
        return corpora.Dictionary(corpus)

    def corpus_to_bow(self, corpus: List[List[str]], dictionary: Dictionary):
        return [dictionary.doc2bow(doc) for doc in corpus]

    def doc_to_bow(self, doc: List[List[str]], dictionary: Dictionary):
        return dictionary.doc2bow(doc)


class TfIdfModel(Model):
    def train(self, topic: str):
        corpus = Corpus.get_corpus(topic)
        dictionary = super().create_dictionary(corpus)
        # convert corpus to BoW format
        bow_corpus = super().corpus_to_bow(corpus, dictionary)

        # fit model
        tfidf = models.TfidfModel(bow_corpus)
        nf = len(dictionary.dfs)
        tfidf_corpus = tfidf[bow_corpus]
        index = similarities.SparseMatrixSimilarity(tfidf_corpus, num_features=nf)

        tfidf.save(super().get_model_path(topic, 'tfidf'))
        index.save(super().get_index_path(topic, 'tfidf'))
        dictionary.save(super().get_dictionary_path(topic, 'tfidf'))

        return corpus, bow_corpus, tfidf_corpus, dictionary

    def predict(self, query: str, topic: str):
        # load model, index and dictionary
        tfidf: models.TfidfModel = models.TfidfModel.load(self.get_model_path(topic, 'tfidf'))
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(
            self.get_index_path(topic, 'tfidf'))
        dictionary: corpora.Dictionary = corpora.Dictionary.load(self.get_dictionary_path(topic, 'tfidf'))

        # convert query to bow format
        query_bow = super().doc_to_bow(query, dictionary)

        # Compute similarity between query and this index
        sims = index[tfidf[query_bow]]
        # Similarity between query and each document sorted

        # Return first 10 most similar documents
        return [(doc_idx, doc_sim) for doc_idx, doc_sim in
                sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:10]]


class Word2VecModel(Model):
    @staticmethod
    def compute_word2vec_similarity(self, model: models.Word2Vec, ws1: List[str], ws2: List[str]):
        """Compute cosine similarity between two sets of words."""
        return model.wv.n_similarity(ws1, ws2)

    def train(self, topic: str):
        pass

    def predict(self, query: str, topic: str):
        corpus = Corpus.get_corpus(topic)

        # Training algorithm: 1 for skip-gram; otherwise CBOW.
        word2vec = models.Word2Vec(corpus, min_count=10, sg=0, window=10)

        # Precompute L2-normalized vectors.
        # If replace is set, forget the original vectors and only keep the normalized ones = saves lots of memory!
        # Note that you cannot continue training after doing a replace. The model becomes effectively read-only = you can call most_similar, similarity etc., but not train.
        word2vec.init_sims(replace=True)

        # Return first 10 most similar documents
        return sorted(
            [(doc_idx, word2vec.wv.n_similarity(query, doc_content)) for doc_idx, doc_content in enumerate(corpus) if
             len(doc_content) != 0], key=lambda x: x[1], reverse=True)[:10]


class LsiModel(Model):
    def train(self, topic: str):
        corpus = Corpus.get_corpus(topic)
        dictionary = super().create_dictionary(corpus)
        bow_corpus = super().corpus_to_bow(corpus, dictionary)
        # TODO: dynamically adjust hyperparameters (num_topics)
        lsi = models.LsiModel(bow_corpus, id2word=dictionary, num_topics=250)
        # transform corpus to LSI space and index it
        index = similarities.MatrixSimilarity(lsi[bow_corpus])

        lsi.save(super().get_model_path(topic, 'lsi'))
        index.save(super().get_index_path(topic, 'lsi'))
        dictionary.save(super().get_dictionary_path(topic, 'lsi'))

    def predict(self, query: str, topic: str):
        lsi = models.LsiModel.load(super().get_model_path(topic, 'lsi'))
        dictionary = corpora.Dictionary.load(super().get_dictionary_path(topic, 'lsi'))
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(
            super().get_index_path(topic, 'lsi'))

        query_bow = super().doc_to_bow(query, dictionary)
        query_lsi = lsi[query_bow]

        # perform a similarity query against the corpus
        sims = index[query_lsi]

        r = [(doc_idx, doc_sim) for doc_idx, doc_sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
        return r[:10]


class LdaModel(Model):
    def train(self, topic: str):
        corpus = Corpus.get_corpus(topic)
        dictionary = super().create_dictionary(corpus)
        bow_corpus = super().corpus_to_bow(corpus, dictionary)

        lda = models.LdaModel(bow_corpus, num_topics=200, passes=5)
        index = similarities.MatrixSimilarity(lda[bow_corpus])

        lda.save(super().get_model_path(topic, 'lda'))
        index.save(super().get_index_path(topic, 'lda'))
        dictionary.save(super().get_dictionary_path(topic, 'lda'))

    def predict(self, query: str, topic: str):
        lda = models.LdaModel.load(super().get_model_path(topic, 'lda'))
        dictionary = corpora.Dictionary.load(super().get_dictionary_path(topic, 'lda'))
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(
            super().get_index_path(topic, 'lda'))

        query_bow = super().doc_to_bow(query, dictionary)
        query_lda = lda[query_bow]

        # perform a similarity query against the corpus
        sims = index[query_lda]

        r = [(doc_idx, doc_sim) for doc_idx, doc_sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
        return r[:10]


class LsiTfidfModel(Model):
    def train(self, topic: str):
        corpus, bow_corpus, tfidf_corpus, dictionary = TfIdfModel().train(topic)

        lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

        # transform corpus to LSI space and index it
        index = similarities.MatrixSimilarity(lsi[tfidf_corpus])

        lsi.save(super().get_model_path(topic, 'lsi-tfidf'))
        index.save(super().get_index_path(topic, 'lsi-tfidf'))
        dictionary.save(super().get_dictionary_path(topic, 'lsi-tfidf'))

    def predict(self, query: str, topic: str):
        lsi = models.LsiModel.load(super().get_model_path(topic, 'lsi-tfidf'))
        dictionary = corpora.Dictionary.load(super().get_dictionary_path(topic, 'lsi-tfidf'))
        index: similarities.SparseMatrixSimilarity = similarities.SparseMatrixSimilarity.load(
            super().get_index_path(topic, 'lsi-tfidf'))

        query_bow = super().doc_to_bow(query, dictionary)
        query_lsi = lsi[query_bow]

        # perform a similarity query against the corpus
        sims = index[query_lsi]

        r = [(doc_idx, doc_sim) for doc_idx, doc_sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
        return r[:10]
