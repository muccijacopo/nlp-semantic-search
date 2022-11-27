import re
from typing import List

import nltk
from nltk.corpus import stopwords

# Download stopwords dataset
# nltk.download('punkt')


class Preprocessing:
    @staticmethod
    def remove_html(content):
        return str(content).replace('<.{1,6}>', '')

    @staticmethod
    def apply_lowercase(content):
        return str(content).lower()

    @staticmethod
    def remove_special_characters(content):
        return re.sub(r'\W', ' ', content)

    @staticmethod
    def apply_decontractions(content: str):
        return content.replace("won't", "will not").str.replace("can\'t", "can not").str.replace("n\'t",
                                                                                                              " not").str.replace(
            "\'re", " are").str. \
            replace("\'s", " is").str.replace("\'d", " would").str.replace("\'ll", " will").str. \
            replace("\'t", " not").str.replace("\'ve", " have").str.replace("\'m", " am")

    @staticmethod
    def remove_stopwords(words: List[str]):
        return [w for w in words if w not in stopwords.words('english')]

    @staticmethod
    def tokenize(sentence: str):
        return nltk.word_tokenize(sentence)

