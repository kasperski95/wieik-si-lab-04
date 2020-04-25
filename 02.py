# For "Olivetti faces" dataset:
# - compare Random Forest with any other classifier (eg. logistic regression)
# - compare Gini vs Entropy
# - check influence of number of trees

import time

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# preprocessing
dataset = fetch_olivetti_faces()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


def check_influence_of_n_trees():
    scores = []
    n_estimators_arr = range(10, 300, 10)
    for n_estimators in n_estimators_arr:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt")
        rfc.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, rfc.predict(X_test)))

    import matplotlib.pyplot as plt

    plt.title("Accuracy in respect to number of trees")
    plt.plot(n_estimators_arr, scores, color="darkcyan")
    plt.savefig("output/02-a.png")


check_influence_of_n_trees()


def compare_Random_Forest_with_any_other_classifier():
    rfc = RandomForestClassifier(n_estimators=410, max_features="sqrt")
    start = time.time()
    rfc.fit(X_train, y_train)
    end = time.time()
    score = accuracy_score(y_test, rfc.predict(X_test))
    print(f"[RandomForestClassifier] time of training: {(end - start):.2f}", "accuracy: ", score)

    lr = LogisticRegression(max_iter=160, verbose=False)
    start = time.time()
    lr.fit(X_train, y_train)
    end = time.time()
    score = accuracy_score(y_test, lr.predict(X_test))
    print(f"[LogisticRegression] time of training: {(end - start):.2f}", "accuracy: ", score)


compare_Random_Forest_with_any_other_classifier()


def compare_Gini_vs_Entropy():
    for i in range(1, 5):
        rfc_gini = DecisionTreeClassifier(criterion="gini")
        rfc_gini.fit(X_train, y_train)
        score = accuracy_score(y_test, rfc_gini.predict(X_test))
        print(f"[{i}][Gini] accuracy: ", score)

        rfc_entropy = DecisionTreeClassifier(criterion="entropy")
        rfc_entropy.fit(X_train, y_train)
        score = accuracy_score(y_test, rfc_entropy.predict(X_test))
        print(f"[{i}][Entropy] accuracy: ", score)


compare_Gini_vs_Entropy()
