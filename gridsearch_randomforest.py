import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold

from gcForest.GCForest import gcForest

from load_data import load_data
from baseline import TitleFinder
from baseline_author import AuthorFinder
from constants import *

def main():

    X, y, tfidf = load_data()

    # Number of Features
    print("Using ", NUM_FEATURES, "Features based on tf-idf")

    # feature selection to make the problem tractable for gcforest
    fs = SelectKBest(k=NUM_FEATURES)
    X = fs.fit_transform(X,y)
    X = np.asarray(X.todense())
    
    X, _, y, _ = train_test_split(X, y, train_size=0.6, random_state=1337, stratify=y)
    
    possibleNumTrees = [200, 400, 800, 1600]

    bestAccuracy = -float("inf")
    bestNumTrees = 0

    folds = StratifiedKFold(random_state=1337)
    for numTrees in possibleNumTrees:
        print("Now testing numTrees=%d" % numTrees)
        scores = []
        for train_index, test_index in folds.split(X, y):
            model = RandomForestClassifier(n_estimators=numTrees)
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        print("Cross validation scores:", scores)
        accuracy = np.mean(scores)

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestNumTrees = numTrees

    print("Best Accuracy = ", bestAccuracy)
    print("Best Num Trees =", bestNumTrees)

if __name__ == '__main__':
    main()