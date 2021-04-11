import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def getError(pred, Y):
    return sum(pred != Y) / float(len(Y))


def generic_clf(x_train, y_train, y_test, x_test, clf):
    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    return getError(pred_train, y_train), \
           getError(pred_test, y_test)

## incomplete
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return getError(pred_train, Y_train), \
           getError(pred_test, Y_test)


if __name__ == '__main__':

    # Read data
    test = pd.read_csv(filepath_or_buffer="/Users/hilaryyyy/Desktop/DogsVsCats/DogsVsCats/DogsVsCats.test", header=None,
                       prefix=None)
    train = pd.read_csv("/Users/hilaryyyy/Desktop/DogsVsCats/DogsVsCats/DogsVsCats.train", header=None, prefix=None)

    y_train = train.iloc[:, 0]
    x_train = train.iloc[:, 1:]

    y_test = test.iloc[:, 0]
    x_test = test.iloc[:, 1:]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    er_tree = generic_clf(y_train, x_train, y_test, x_test, clf_tree)

    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    x_range = range(10, 410, 10)
    for i in x_range:
        er_i = adaboost_clf(y_train, x_train, y_test, x_test, i, clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])