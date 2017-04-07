from load_data import load_data
from constants import *
import random
import numpy as np

from sklearn.base import BaseEstimator

class AuthorFinder(BaseEstimator):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.indices = {}

        author_words = ["andy", "weir", "donna", "tartt", "james", "el", "flynn", "john", "greene", "laura", "hillenbrand", "paula", "hawkins", "suzanne", "collins"]
        for author in author_words:
            if author in self.feature_names:
                self.indices[author] = self.feature_names.index(author)

    def fit(self, X, y):
        """
        does nothing for this simple baseline
        """
        return self

    def predict(self, X):
        n,d = X.shape
        y = np.zeros(n)

        for row in range(n):
            data = np.asarray(X[row,:].todense()).reshape([-1])
            if data[self.indices["andy"]] > 0 or data[self.indices["weir"]] > 0:
                y[row] = 0
            elif data[self.indices["donna"]] > 0 or data[self.indices["tartt"]] > 0:
                y[row] = 1
            elif data[self.indices["el"]] > 0 or data[self.indices["james"]] >0:
                y[row] = 2
            elif data[self.indices["flynn"]] > 0:
                y[row] = 3
            elif data[self.indices["john"]] > 0 or data[self.indices["greene"]] > 0:
                y[row] = 4
            elif data[self.indices["laura"]] > 0 or data[self.indices["hillenbrand"]] > 0:
                y[row] = 5
            elif data[self.indices["paula"]] > 0 or data[self.indices["hawkins"]] > 0:
                y[row] = 6
            elif data[self.indices["suzanne"]] > 0 or data[self.indices["collins"]] > 0:
                y[row] = 7
            else:
                y[row] = random.randint(0,7)
                
        return y
            
            

    


