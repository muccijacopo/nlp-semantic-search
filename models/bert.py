from sentence_transformers import SentenceTransformer, util, models, InputExample, losses
import torch
from torch.utils.data import DataLoader

from models import Model
from corpus import Corpus


class MiniLMModel(Model):
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
        corpus = Corpus.get_corpus(topic, tokenize=False)

        model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)

        return [(idx.item(), score.item()) for score, idx in zip(top_results[0], top_results[1])]


class FineTunedBertModel(Model):
    def train(self, topic: str):
        # TODO: move model training here and implement saving
        pass

    def predict(self, query: str, topic: str):

        corpus = Corpus.get_corpus(topic, tokenize=False)

        transformer_model = models.Transformer('bert-base-uncased', max_seq_length=256)
        pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                                   activation_function=torch.nn.Tanh())

        model = SentenceTransformer(modules=[transformer_model, pooling_model])

        # train_examples = [InputExample(texts=[corpus])]
        train_examples = [InputExample(texts=[doc for doc in corpus], label=1)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=10)
        cosine_similarity_loss = losses.CosineSimilarityLoss(model)

        # Tune model
        model.fit(train_objectives=[(train_dataloader, cosine_similarity_loss)], epochs=1)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)

        return [(idx.item(), score.item()) for score, idx in zip(top_results[0], top_results[1])]


