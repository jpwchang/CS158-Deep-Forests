from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from glob import iglob
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from load_data import load_data
from constants import *



def main():
    X, y, tfidf = load_data()
    fnames = tfidf.get_feature_names()
    classes = np.unique(y)
    for book in classes:
        book_indices = np.where(y==book)[0]
        book_features = np.sum(X[book_indices, :], axis=0)
        top_feature_indices = np.argpartition(book_features, -15)[-15:]
        top_feature_values = book_features[top_feature_indices]

        plt.figure()
        plt.bar(np.arange(15), top_feature_values)
        plt.xlabel("Word")
        plt.ylabel("Tfidf score")
        plt.title("Top words in book %d", book)
        plt.xticks(np.arange(15), fnames[top_feature_indices])
        plt.show()

    return

if __name__ == '__main__':
    main()
