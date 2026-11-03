import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X = StandardScaler().fit_transform(X)

scores = []

for seed in range(30):

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        learning_rate_init=0.001,
        random_state=seed
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    scores.append(
        accuracy_score(y_test, pred)
    )

print("Mean Accuracy =", np.mean(scores))
print("Std =", np.std(scores, ddof=1))
