import re
import difflib
from collections import Counter, defaultdict
import nltk


class LemmaTokenizer:
    def __init__(self):
        self._lemmatize = nltk.wordnet.WordNetLemmatizer().lemmatize
        self._tokenize = nltk.word_tokenize
        self._tagging = nltk.pos_tag
        self._stemmer = nltk.stem.PorterStemmer()
        self._wordnet = nltk.corpus.wordnet
        self._suffix = {"ied": "y", "ies": "y", "ed": "", "ing": ""}
        self._tagmaps = {"JJ": self._wordnet.ADJ, "RB": self._wordnet.ADV,
                         "VB": self._wordnet.VERB}
        
    def __call__(self, text):
        for p in ["/", "\n"]:
            text = text.replace(p, " ")
        lemmas = []
        for word, tag in self._tagging(self._tokenize(text)):
            tag = self._tagmaps.get(tag[:2], self._wordnet.NOUN)
            tag = self._wordnet.VERB
            word = self._lemmatize(word, tag)
            flag = word[0] == word[0].upper()
            word = self._stemmer.stem(word)
            if flag:
                word = word.capitalize()
            for x, y in self._suffix.items():
                if word.endswith(x):
                    temp = word[:-len(x)] + y
                    if len(self._wordnet.synsets(temp)) > 0:
                        word = temp
                        break
            lemmas.append(word)
        return lemmas

tokenizer = LemmaTokenizer()


import string
stop_words = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))
def question_token(ngram1):
    
    if isinstance(ngram1, str):
        ngram1 = tokenizer(ngram1)
    ngram1 = {_.lower() for _ in ngram1 if _.lower() not in stop_words and _.isalpha() and _ != _.capitalize()}
    
    return ngram1

