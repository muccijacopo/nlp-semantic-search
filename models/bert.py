from sentence_transformers import SentenceTransformer, util
import torch

from models import Model
from corpus import Corpus


class BERTModel(Model):
    def train(self, topic: str):
        corpus = Corpus.get_corpus('beer', tokenize=False)

        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        # query_embedding = model.encode(query, convert_to_tensor=True)
        #
        # cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        # top_results = torch.topk(cos_scores, k=10)

        # return [(idx.item(), score.item()) for score, idx in zip(top_results[0], top_results[1])]

    def predict(self, query: str, topic: str):
        corpus = Corpus.get_corpus('beer', tokenize=False)

        model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)

        return [(idx.item(), score.item()) for score, idx in zip(top_results[0], top_results[1])]
