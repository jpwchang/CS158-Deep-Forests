from sklearn.feature_extraction.text import CountVectorizer
from glob import iglob
from bs4 import BeautifulSoup
import pandas as pd

DATA_PATH = "/home/jonathan/Downloads/amazon_book_reviews/"

texts = []
for filename in iglob(DATA_PATH + "*.csv"):
    csv_table = pd.read_csv(filename, delimiter='\t')
    csv_table.columns = ['rating', 'url', 'title', 'html']
    texts += [BeautifulSoup(h, 'lxml').text for h in csv_table.html]

cv = CountVectorizer()
cv.fit(texts)
print(len(cv.vocabulary_.keys()))