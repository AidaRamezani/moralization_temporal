import re

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

STOP = stopwords.words("english")
STOP += [
    "like",
    "gone",
    "did",
    "going",
    "would",
    "could",
    "get",
    "in",
    "up",
    "may",
    "wanted",
    "actually",
    "likely",
    "possibly",
    "want",
    "make",
    "my",
    "someone",
    "sometimes",
    "would",
    "want_to",
    "s",
    "its",
    "one",
    "something",
    "sometimes",
    "everybody",
    "somebody",
    "could",
    "be",
    "my",
    "you",
    "it",
    "its",
    "your",
    "i",
    "he",
    "she",
    "his",
    "her",
    "they",
    "them",
    "their",
    "our",
    "we",
]


def sentensize_articles(article_text):
    all_sentences = []
    for article in article_text:
        sentences = nltk.sent_tokenize(article)
        all_sentences += sentences
    return all_sentences


class MyTokenizer:
    lemmatizer = WordNetLemmatizer()

    def lemmatize_tokens(self, token_list: list[str]):
        token_list = list(map(self.clean_sentence, token_list))

        token_list = [self.lemmatizer.lemmatize(t) for t in token_list]
        return token_list

    def clean_tokens(self, token_list: list[str], lemmatize = False):
        token_list = self.remove_stop(token_list)  # no punctuations, no stop words
        return token_list

    def clean_tokens_batch(self, token_lists: list[list[str]], lemmatize = False):
        token_lists = list(
            map(self.clean_tokens, token_lists, [lemmatize for t in token_lists]),
        )
        return token_lists

    def remove_stop(self, token_list):
        tokens = [x for x in token_list if x not in STOP and x != ""]
        return tokens

    def clean_sentence(self, sentence):
        text = re.sub("[^a-zA-Z]", "", sentence)
        text = text.lower()
        return text

    def simple_tokenize(self, sentence: str, lemmatize=False) -> list:

        tokens = sentence.split()
        return tokens

    def tokenize(self, sentence: str, lemmatize=False) -> list:
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def tokenize_batch(self, sentences, lemmatize, method):

        assert method in ["standard", "simple"]
        if method == "standard":
            tokens_list = list(
                map(self.tokenize, sentences, [lemmatize for t in sentences]),
            )

        else:
            tokens_list = list(
                map(self.simple_tokenize, sentences, [lemmatize for t in sentences]),
            )
        lemmatized_tokens = list(map(self.lemmatize_tokens, tokens_list))
        tokens_list_clean = list(
            map(self.clean_tokens, lemmatized_tokens, [lemmatize for t in sentences]),
        )
        return tokens_list, lemmatized_tokens, tokens_list_clean
