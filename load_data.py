from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix
from glob import iglob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import pickle

from constants import *

def load_data():
    # load the data from file if possible
    if os.path.isfile("X_cached.npz") and os.path.isfile("y_cached.npy") and os.path.isfile("tfidf_cached.pickle"):
        loader = np.load("X_cached.npz")
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        y = np.load("y_cached.npy")
        tfidf_file = open("tfidf_cached.pickle", "rb")
        tfidf = pickle.load(tfidf_file)
        tfidf_file.close()
        return X, y, tfidf

    # otherwise generate from scratch
    texts = []
    labels = []
    cur_text = 0
    for filename in iglob(os.path.join(DATA_PATH, "*.csv")):
        csv_table = pd.read_csv(filename, delimiter='\t')
        csv_table.columns = ['rating', 'url', 'title', 'html']
        csv_table = csv_table.head(10000)
        texts += [BeautifulSoup(h, 'lxml').text for h in csv_table.html]
        labels += [cur_text for i in csv_table.html]
        cur_text += 1
        print("Finished processing file:" + filename)

    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(texts)
    y = np.array(labels)
    # save data to file for future use
    np.savez("X_cached.npz", data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
    np.save("y_cached.npy", y)
    tfidf_file = open("tfidf_cached.pickle", "wb")
    pickle.dump(tfidf, tfidf_file)
    tfidf_file.close()
    return X, y, tfidf

if __name__ == '__main__':
    load_data()
