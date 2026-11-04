import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# GFNN
# =====================================

class GFNN:

    def __init__(
        self,
        initial_neurons=8,
        growth_step=2,
        max_neurons=66,
        threshold=0.05
    ):

        self.initial_neurons = initial_neurons
        self.growth_step = growth_step
        self.max_neurons = max_neurons
        self.threshold = threshold

        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        best_acc = -1.0
        neurons = self.initial_neurons

        while neurons <= self.max_neurons:

            model = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                max_iter=500,
                random_state=0
            )

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)

            acc = accuracy_score(
                y_val,
                y_pred_val
            )

            if acc > best_acc:

                best_acc = acc

                self.model = model

            if (1.0 - acc) <= self.threshold:
                break

            neurons += self.growth_step

    def predict(self, X):

        return self.model.predict(X)

# =====================================
# DATA
# =====================================

X, y = load_iris(return_X_y=True)

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =====================================
# STORAGE
# =====================================

accs = []
precs = []
recs = []
f1s = []

print("START")

# =====================================
# 30 RUNS
# =====================================

for seed in range(30):

    print("RUN", seed + 1)

    X_train_full, X_test, y_train_full, y_test = \
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=seed
        )

    X_train, X_val, y_train, y_val = \
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.25,
            stratify=y_train_full,
            random_state=seed
        )

    model = GFNN()

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    rec = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)

# =====================================
# RESULTS
# =====================================

print()
print("=" * 50)
print("IRIS RESULTS")
print("=" * 50)

print("Runs =", len(accs))

print(
    "Accuracy Mean =",
    np.mean(accs)
)

print(
    "Accuracy Std =",
    np.std(accs)
)

print(
    "Precision Mean =",
    np.mean(precs)
)

print(
    "Recall Mean =",
    np.mean(recs)
)

print(
    "F1 Mean =",
    np.mean(f1s)
)

print("END")
print("Length GFNN:", len(gfnn_accs))
print("Length Baseline:", len(baseline_accs))

print("GFNN:", gfnn_accs[:5])
print("Baseline:", baseline_accs[:5])
print("END")