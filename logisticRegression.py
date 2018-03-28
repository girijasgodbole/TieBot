from sklearn.feature_extraction.text import CountVectorizer
import numpy
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier


class LogisticRegError(Exception):
    "Defines error of logistic reg learner"


class SVMClassifier:

    def __init__(self):
        pass

    def vectorize(self, corpus):
        if not isinstance(corpus, list):
            raise(LogisticRegError('Corpus must be of type:list()'))
        vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer='word')
        X = vectorizer.fit_transform(corpus).toarray()
        return X.reshape(X.shape[0], len(X[0])), vectorizer


    def svmClassifer(self, X, y):
        return OneVsOneClassifier(SVC(C=1, cache_size=400, coef0=0.0, degree=5, gamma='auto',
                                      kernel='rbf', max_iter=-1, shrinking=True, tol=.01, verbose=False), -1).fit(X, y)

    def run(self, corpus, query):
        vectorize_corpus, vectorizer = self.vectorize(corpus)
        vectorize_query = vectorizer.transform(query).toarray()
        Y = range(0, vectorize_corpus.shape[0])
        print 'X: ', vectorize_corpus
        print 'Y: ', Y
        clf = self.svmClassifer(vectorize_corpus, Y)
        print clf.predict(vectorize_query)


def main():
    corpus = ['Hey hey hey lets go get lunch today',
              'Did you go home',
              'Hey!!! I need a favor',
              'Hey lets go get a drink tonight']
    query = ['are you going home']
    svmc = SVMClassifier()
    svmc.run(corpus, query)

if __name__=='__main__':
    main()