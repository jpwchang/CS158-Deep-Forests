import sys
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest

from gcForest.GCForest import gcForest

from load_data import load_data
from baseline import TitleFinder
from baseline_author import AuthorFinder
from constants import *

def main():

    X,y,tfidf = load_data()

    # feature selection to make the problem tractable for gcforest
    if sys.argv[1] != "baseline" and sys.argv[1] != "author":
        fs = SelectKBest(k=NUM_FEATURES)
        X = fs.fit_transform(X,y)
        X = np.asarray(X.todense())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1337, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=800, n_jobs=20)
    gc = gcForest(shape_1X=NUM_FEATURES, n_cascadeRF=6, n_cascadeRFtree=100, n_jobs=20)

    print("Training random forest...")
    rf.fit(X_train,y_train)
    y_train_pred = rf.predict(X_train)
    print("Random forest accuracy on training set:", accuracy_score(y_train, y_train_pred))
    print("Training deep forest...")
    gc.fit(X_train,y_train)
    y_train_pred = gc.predict(X_train)
    print("Deep forest accuracy on training set:", accuracy_score(y_train, y_train_pred))

    # Use bootstrapping to compute CI's and perform t-test
    print("Evaluating models using bootstrapping...")
    n, d = X_test.shape
    y_pred_rf = rf.predict(X_test)
    y_pred_gc = gc.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    gc_acc = accuracy_score(y_test, y_pred_gc)
    rf_scores = []
    gc_scores = []
    for t in range(1000):
        rand = np.random.randint(0, n, n)
        X_bs, y_bs = X_test[rand], y_test[rand]
        y_pred_rf = rf.predict(X_bs)
        y_pred_gc = gc.predict(X_bs)
        rf_scores.append(accuracy_score(y_bs, y_pred_rf))
        gc_scores.append(accuracy_score(y_bs, y_pred_gc))
    lower_rf = np.percentile(rf_scores, 2.5)
    upper_rf = np.percentile(rf_scores, 97.5)
    lower_gc = np.percentile(gc_scores, 2.5)
    upper_gc = np.percentile(gc_scores, 97.5)
    print("Random Forest accuracy: %.3f (%.3f, %.3f)" % (rf_acc, lower_rf, upper_rf))
    print("Deep Forest accuracy: %.3f (%.3f, %.3f)" % (gc_acc, lower_gc, upper_gc))
    _, p = stats.ttest_rel(rf_scores, gc_scores)
    print("p =", p)

if __name__ == '__main__':
    main()
