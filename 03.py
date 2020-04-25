# Experiment with face recreation using random forest. Save results.

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from os import path

# preprocessing
dataset = fetch_olivetti_faces()
X = dataset.data
y = dataset.target
X_train_whole, X_test_whole = train_test_split(X, test_size=0.1, random_state=0)

# split upper and lower part of a face
X_train = X_train_whole[:, :2048]  # 4096/2
Y_train = X_train_whole[:, 2048:]

X_test = X_test_whole[:, :2048]
Y_test = X_test_whole[:, 2048:]

print("Training samples: ", X_train.shape[0])
print("Testing samples: ", X_test.shape[0])

random_forest = ExtraTreesRegressor(n_estimators=100, max_features=20, random_state=0)

if path.exists("output/03-model.pkl"):
    print("Loading model")
    with open("output/03-model.pkl", "rb") as f:
        random_forest = pickle.load(f)
else:
    print("Training and saving the model")
    random_forest.fit(X_train, Y_train)
    with open("output/03-model.pkl", "wb") as f:
        pickle.dump(random_forest, f)


def plot_face(id):
    plt.subplot("121")
    original_face = np.hstack((X_test[id].reshape(1, -1), Y_test[id].reshape(1, -1))).reshape(64, 64)
    plt.imshow(original_face, cmap=plt.cm.gray, interpolation="nearest")

    plt.subplot("122")
    guessed_face = np.hstack((X_test[id].reshape(1, -1), random_forest.predict(X_test[id].reshape(1, -1)))).reshape(64, 64)
    plt.imshow(guessed_face.reshape(64, 64), cmap=plt.cm.gray, interpolation="nearest")

    plt.savefig(f"output/03-faces/03-{id}.png")
    plt.close()


for i in range(0, X_test.shape[0]):
    plot_face(i)
