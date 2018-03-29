from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd

class LogisticRegError(Exception):
    "Defines error of logistic reg learner"


class SVMClassifier:
    def __init__(self):
        pass

    def tokenize(self, corpus):
        if not isinstance(corpus, dict):
            raise(LogisticRegError('Corpus must be of type:dict()'))
        # tokenize each sentence and stem each word
        stemmer = LancasterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        for word in corpus:
            corpus[word] = corpus[word].lower()
            corpus[word] = tokenizer.tokenize(corpus[word])
        return corpus

    def getFeaturesByName(self, corpus):
        if not isinstance(corpus, list):
            raise(LogisticRegError('Corpus must be of type:list()'))
        corpus = corpus.values()
        newCorpus = list()
        for eachList in corpus:
            newCorpus.extend(eachList)
        corpus = newCorpus
        corpus = list(set(corpus))

    def BOW(self, corpus):
        if not isinstance(corpus, dict):
            raise(LogisticRegError('Corpus must be of type:dict()'))
        tokenDict = self.tokenize(corpus)
        getWordFeatures = self.getFeaturesByName(tokenDict)
        vectorDataFrame = pd.DataFrame()
        for word in corpus:
            vectorDataFrame[word] = np.array([1 if x in getWordFeatures else 0 for x in corpus[word]])
        return vectorDataFrame

    def svmClassifer(self, X, y):
        return OneVsOneClassifier(SVC(C=1, cache_size=400, coef0=0.0, degree=5, gamma='auto',
                                      kernel='rbf', max_iter=-1, shrinking=True, tol=.01, verbose=False), -1).fit(X, y)

    def LogisticRegClassifier(self, X, y):
        return LogisticRegression().fit(X, y)


    def run(self, corpus, query):
        corpus = self.tokenize(corpus)
        corpus = self.BOW(corpus)
        print corpus

        vectorize_corpus, vectorizer = self.vectorize(corpus)
        vectorize_query = vectorizer.transform(query).toarray()
        Y = range(0, vectorize_corpus.shape[0])
        clf = self.LogisticRegClassifier(vectorize_corpus, Y)
        print clf.predict(vectorize_query)

    def sigmoidDerivative(self, X):
        return X*(1-X)




def main():
    corpus ={'1':"Hey hey hey let's go get lunch today?",
              '2':'Did you go home',
              '3':'Hey!!! I need a favor',
              '4':'Hey lets go get a drink tonight'}
    query = ['are you going home']
    svmc = SVMClassifier()
    svmc.run(corpus, query)


if __name__=='__main__':
    main()