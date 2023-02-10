from abc import abstractmethod
from typing import List

from sentence_transformers import SentenceTransformer, util, models, InputExample, losses
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model, pipeline, set_seed, \
    QuestionAnsweringPipeline

from models import Model
from corpus import Corpus


class MiniLMModel(Model):
    def train(self, topic: str):
        corpus = Corpus.get_corpus(topic, tokenize=False)

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


class MultiQAMiniLMWithTorch(Model):

    def __init__(self, topic):
        self.corpus = Corpus.get_corpus(topic, tokenize=False)[:100]
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    @classmethod
    def mean_pooling(cls, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Encode text
    def encode(self, docs):
        # Tokenize sentences
        encoded_input = self.tokenizer(docs, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        # TODO: use Dataloader for batching
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def train(self, topic: str):
        pass

    def predict(self, query: str, topic: str):
        corpus_embeddings = self.encode(self.corpus)
        query_embeddings = self.encode(query)

        # Compute dot score between query and all document embeddings
        # TODO: try cosine similarity
        scores = torch.mm(query_embeddings, corpus_embeddings.transpose(0, 1))[0].cpu().tolist()
        # Combine docs & scores
        # TODO: construct matrix similarity
        # docs_similarity = list(zip(self.corpus, scores))
        # docs_similarity = enumerate(scores)

        # Sort docs by similarity score
        docs_similarity_sorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return docs_similarity_sorted[:10]


class DistilBertModel(Model):

    MODEL_NAME = "sentence-transformers/multi-qa-distilbert-cos-v1"

    def __init__(self):
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    @classmethod
    def get_tensor_file_name(cls, topic: str):
        return f'models/stored_models/{topic}_distilbert.pt'

    @classmethod
    def mean_pooling(cls, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, docs):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def train(self, topic: str):
        corpus = Corpus.get_corpus(topic, tokenize=False)[:1000]
        corpus_dataset = TorchDataset(corpus)
        corpus_dataloader = DataLoader(corpus_dataset, batch_size=100)
        corpus_embeddings = None
        for corpus_batch in corpus_dataloader:
            if corpus_embeddings is None:
                corpus_embeddings = self.encode(corpus_batch)
            else:
                corpus_embeddings = torch.cat((corpus_embeddings, self.encode(corpus_batch)), 0)

        print(corpus_embeddings)
        torch.save(corpus_embeddings, self.get_tensor_file_name(topic))

    def predict(self, query: str, topic: str):
        # corpus_embeddings = self.encode(self.corpus)
        query_embeddings = self.encode(query)

        corpus_embeddings = torch.load(self.get_tensor_file_name(topic))

        # Compute dot score between query and all document embeddings
        scores = torch.mm(query_embeddings, corpus_embeddings.transpose(0, 1))[0].cpu().tolist()

        # Sort docs by similarity score
        docs_similarity_sorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return docs_similarity_sorted[:10]


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


class GenerativeModel:
    @abstractmethod
    def generate(self, question: str, context: str):
        pass


class QuestionAnsweringGPT2Model(GenerativeModel):
    model_name = "distilgpt2"

    def generate(self, question: str, context: str):
        print(question)
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        text_generation_pipeline = pipeline('text-generation', model=self.model_name)
        # model = GPT2Model.from_pretrained('gpt2')

        # encoded_input = tokenizer(text, return_tensors='pt')
        # output = model(**encoded_input)
        outputs = text_generation_pipeline(question, num_return_sequences=10, max_length=100)
        # print(output)

        return outputs


class QuestionAnsweringDistilbertModel(GenerativeModel):
    def generate(self, question: str, context: List[str]):
        # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        # model = DistilBertModel.from_pretrained('distilbert-base-cased-distilled-squad')
        question_answerer_pipeline = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
        # print(question, context)
        # inputs = tokenizer(question, text, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = model(**inputs)
        #
        # print(outputs)
        # print(res)
        # res = QuestionAnsweringGPT2Model().generate(res[0]['answer'], '')\

        res = question_answerer_pipeline(question=question, context=context, top_k=10)
        return sorted(res, key=lambda x: x['score'], reverse=True)


class TorchDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]