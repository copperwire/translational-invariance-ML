from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

import numpy as np


def find_score(N):
    # producing training data
    domain_edges = np.arange(0, 500)
    slopes = np.linspace(-100, 100, int(1e5))
    intercepts = np.linspace(-100, 100, int(1e5))
    full_dom = np.linspace(-100, 100, 500)

    data = []
    targets = []
    n_datapoints = int(N)
    i = 0

    while i < n_datapoints:
        start = np.random.choice(domain_edges)
        stop = start + 40

        a = np.random.choice(slopes)
        b = np.random.choice(intercepts)
        datapoint = np.array(full_dom)
        datapoint[start:stop] = datapoint[start:stop] * a + b

        data.append(datapoint)
        targets.append(a)
        i += 1
    data = np.array(data)
    targets = np.array(targets)

    # training ANN to find slopes

    X_train, X_test, Y_train, Y_test = train_test_split(data, targets)

    ANN = MLPRegressor()
    ANN.fit(X_train, Y_train)
    predicted = ANN.predict(X_test)
    score = explained_variance_score(Y_test, predicted)
    return score


scores = []
N_list = np.logspace(3, 5, num=10)

for N_val in N_list:
    scores.append(find_score(N_val))

plt.plot(N_list, scores, "x-")
plt.xlabel("Number of generated samples")
plt.ylabel("Explained variance score")
plt.savefig("explained_var.png")
