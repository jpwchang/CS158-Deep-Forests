from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from glob import iglob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import cPickle as pickle

from constants import *

def load_data():
    # load the data from file if possible
    if os.path.isfile("X_cached.npy") and os.path.isfile("y_cached.npy") and os.path.isfile("tfidf_cached.pickle"):
        X = np.load("X_cached.npy")
        y = np.load("y_cached.npy")
        tfidf_file = open("tfidf_cached.pickle", "r")
        tfidf = pickle.load(tfidf_file)
        tfidf_file.close()
        return X, y

    # otherwise generate from scratch
    texts = []
    labels = []
    cur_text = 0
    for filename in iglob(os.path.join(DATA_PATH, "*.csv")):
        csv_table = pd.read_csv(filename, delimiter='\t')
        csv_table.columns = ['rating', 'url', 'title', 'html']
        texts += [BeautifulSoup(h, 'lxml').text for h in csv_table.html]
        labels += [cur_text for i in csv_table.html]
        cur_text += 1
        print("Finished processing file:" + filename)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    y = np.array(labels)
    # save data to file for future use
    np.save("X_cached.npy", X)
    np.save("y_cached.npy", y)
    tfidf_file = open("tfidf_cached.pickle", "w")
    pickle.dump(tfidf, tfidf_file)
    tfidf_file.close()
    return X, y, tfidf

if __name__ == '__main__':
    load_data()
