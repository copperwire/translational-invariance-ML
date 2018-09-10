from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

import numpy as np

# producing training data
domain = np.linspace(0, 10, 100)
slopes = np.linspace(-100, 100, int(1e5))
intercepts = np.linspace(-100, 100, int(1e5))

data = []
targets = []
n_datapoints = int(1e3)

for i in range(n_datapoints):
    a = np.random.choice(slopes)
    b = np.random.choice(intercepts)

    data.append(a*domain + b)
    targets.append(a)

data = np.array(data)
targets = np.array(targets)

# training ANN to find slopes

X_train, X_test, Y_train, Y_test = train_test_split(data, targets)

ANN = MLPRegressor()
ANN.fit(X_train, Y_train)
predicted = ANN.predict(X_test)
score = explained_variance_score(Y_test, predicted)
print(score)
