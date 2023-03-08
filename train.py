from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
import numpy as np

class MostFrequentLabelBaseline(BaseEstimator):
    '''
    This is a baseline classifier that always predicts the most frequent label
    '''

    def __init__(self):
        self.most_frequent_label = None

    def fit(self, X, y):
        """
        Determine the most frequent label from y
        """
        labels, counts = np.unique(y,return_counts=True)
        max_label = np.argmax(counts)
        self.most_frequent_label = labels[max_label]

    def predict(self, X):
        return np.full(len(X), self.most_frequent_label)

class KMeansBaseline(KMeans):
    '''
    This is a baseline classifier that predicts using K-nearest neighbors
    '''

    def __init__(self, labels):
        self.n_clusters = len(np.unique(labels))
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, X, y):
        """
        Determine the most frequent label from y
        """
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_

    def predict(self, X):
        return self.kmeans.predict(X)