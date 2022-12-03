from typing import List
from gensim import corpora, similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec, LsiModel


class Embeddings:

    @staticmethod
    def create_dictionary(corpus: List[List[str]]):
        return corpora.Dictionary(corpus)

    @staticmethod
    def corpus_to_bag_of_words(corpus: List[List[str]], dictionary: Dictionary):
        return [dictionary.doc2bow(doc) for doc in corpus]

    @staticmethod
    def tfidf_similarity(query: List[str], corpus: List[List[str]]):
        dictionary = Embeddings.create_dictionary(corpus)
        # convert corpus to BoW format
        bow_corpus = Embeddings.corpus_to_bag_of_words(corpus, dictionary)

        # fit model
        tfidf = TfidfModel(bow_corpus)

        # transform the whole corpus via TfIdf and store in index matrix
        nf = len(dictionary.dfs)
        index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=nf)

        query_bow = dictionary.doc2bow(query)

        # Compute similarity between query and this index
        sims = index[tfidf[query_bow]]
        # Similarity between query and each document sorted
        res = [e for e in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
        return res[:10]

    @staticmethod
    def word2vec_similarity(query: List[str], corpus: List[List[str]]):

        # Training algorithm: 1 for skip-gram; otherwise CBOW.
        word2vec = Word2Vec(corpus, min_count=10, sg=0, window=10)

        # Precompute L2-normalized vectors.
        # If replace is set, forget the original vectors and only keep the normalized ones = saves lots of memory!
        # Note that you cannot continue training after doing a replace. The model becomes effectively read-only = you can call most_similar, similarity etc., but not train.
        word2vec.init_sims(replace=True)

        """Compute cosine similarity between two sets of words."""
        r = sorted([(d, word2vec.wv.n_similarity(query, d)) for d in corpus if len(d) != 0], key=lambda x: x[1], reverse=True)
        return r[:10]

    @staticmethod
    def latent_semantic_indexing(query: List[str], corpus: List[List[str]]):
        dictionary = Embeddings.create_dictionary(corpus)
        query_bow = dictionary.doc2bow(query)
        bow_corpus = Embeddings.corpus_to_bag_of_words(corpus, dictionary)

        # fit model
        lsi = LsiModel(bow_corpus, id2word=dictionary, num_topics=2)

        # transform corpus to LSI space and index it
        index = similarities.MatrixSimilarity(lsi[bow_corpus])
        # convert the query to LSI space
        vec_lsi = lsi[query_bow]

        # perform a similarity query against the corpus
        sims = index[vec_lsi]
        # sort similarities
        sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:10]

        return[(corpus[doc_idx], doc_sim) for doc_idx, doc_sim in sims]
