from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from glob import iglob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

from load_data import load_data
from constants import *



def main():
    X, y, tfidf = load_data()
    

    return

if __name__ == '__main__':
    main()
