import re

import gensim.utils
import nltk
from typing import List
from pandas import Series
from nltk.corpus import stopwords


# Download stopwords dataset
# nltk.download('punkt')


class CustomPreprocessing:
    @staticmethod
    def remove_html(content):
        return re.sub(r'<.{1,6}>', ' ', str(content))

    @staticmethod
    def apply_lowercase(content):
        return str(content).lower()

    @staticmethod
    def remove_special_characters(content):
        return re.sub(r'\W', ' ', str(content))

    @staticmethod
    def apply_decontractions(content: str):
        return str(content) \
            .replace("won't", "will not") \
            .replace("can\'t", "can not") \
            .replace("n\'t", " not") \
            .replace("\'re", " are") \
            .replace("\'s", " is") \
            .replace("\'d", " would") \
            .replace("\'ll", " will") \
            .replace("\'t", " not") \
            .replace("\'ve", " have") \
            .replace("\'m", " am")

    @staticmethod
    def remove_stopwords(words: List[str]):
        return [w for w in words if w not in stopwords.words('english')]

    @staticmethod
    def tokenize_sentence(sentence: str):
        return nltk.word_tokenize(sentence)

    @staticmethod
    def simple_preprocess(doc: str, tokenize=True, lowercase=True, special_characters=True):
        if lowercase:
            doc = CustomPreprocessing.apply_lowercase(doc)
        if special_characters:
            doc = CustomPreprocessing.remove_html(doc)
            doc = CustomPreprocessing.remove_special_characters(doc)
        if tokenize:
            tokens = CustomPreprocessing.tokenize_sentence(doc)
            tokens = CustomPreprocessing.remove_stopwords(tokens)
            return tokens
        else:
            return doc

    @staticmethod
    def preprocess_query(s: str, tokenize=True):
        s = CustomPreprocessing.apply_lowercase(s)
        s = CustomPreprocessing.remove_special_characters(s)
        if tokenize:
            return CustomPreprocessing.remove_stopwords(CustomPreprocessing.tokenize_sentence(s))
        else:
            return s


class GensimPreprocessing:
    @staticmethod
    def simple_preprocess(doc):
        return gensim.utils.simple_preprocess(doc)
