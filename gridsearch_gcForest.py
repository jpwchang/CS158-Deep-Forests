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
    
    possibleNumTrees = [25, 50, 100, 200]
    possibleNumForests = [2, 4, 6]

    bestAccuracy = -float("inf")
    bestNumTrees = 0
    bestNumForests = 0

    folds = StratifiedKFold(random_state=1337)
    for numForests in possibleNumForests:
        for numTrees in possibleNumTrees:
            print("Now testing numForests=%d, numTrees=%d" % (numForests, numTrees))
            scores = []
            for train_index, test_index in folds.split(X):
                model = gcForest(shape_1X=NUM_FEATURES, n_cascadeRF=numForests, n_cascadeRFtree=numTrees, n_jobs=-1)
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(y_test)
                scores.append(accuracy_score(y_test, y_pred))
            print("Cross validation scores:", scores)
            accuracy = np.mean(scores)

            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestNumTrees = numTrees
                bestNumForests = numForests
    
    print("Best Accuracy = ", bestAccuracy)
    print("best Num Forests =". bestNumForests)
    print("Best Num Trees =", bestNumTrees)

if __name__ == '__main__':
    main()