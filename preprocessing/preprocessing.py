import re
import nltk
from typing import List
from pandas import Series
from nltk.corpus import stopwords


# Download stopwords dataset
# nltk.download('punkt')


class CustomPreprocessing:
    @staticmethod
    def remove_html(content):
        return str(content).replace('<.{1,6}>', '')

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
    def tokenize_list_of_sentences(sentences):
        r: List[List[str]] = []
        for s in sentences:
            tokens = CustomPreprocessing.tokenize_sentence(s)
            tokens = CustomPreprocessing.remove_stopwords(tokens)
            r.append(tokens)
        return r

    @staticmethod
    def reprocess_title(series: Series):
        series.apply(CustomPreprocessing.apply_lowercase)
        series.apply(CustomPreprocessing.remove_special_characters)
        series.apply(CustomPreprocessing.remove_html)
        return series

    @staticmethod
    def preprocess_query(s: str):
        s = CustomPreprocessing.apply_lowercase(s)
        s = CustomPreprocessing.remove_special_characters(s)
        s = CustomPreprocessing.remove_stopwords(CustomPreprocessing.tokenize_sentence(s))
        return s
