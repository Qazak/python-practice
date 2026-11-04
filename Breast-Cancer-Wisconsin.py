import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# GFNN
# ==================================================

class GFNN:

    def __init__(
        self,
        initial_neurons=8,
        growth_step=4,
        max_neurons=64,
        threshold=0.02
    ):

        self.initial_neurons = initial_neurons
        self.growth_step = growth_step
        self.max_neurons = max_neurons
        self.threshold = threshold

        self.model = None

        self.growth_events = 0
        self.final_neurons = initial_neurons

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
                activation="tanh",
                solver="adam",
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=0
            )

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_val
            )

            acc = accuracy_score(
                y_val,
                pred
            )

            if acc > best_acc:

                best_acc = acc

                self.model = model

                self.final_neurons = neurons

            discrepancy = 1.0 - acc

            if discrepancy <= self.threshold:
                break

            neurons += self.growth_step
            self.growth_events += 1

    def predict(self, X):

        return self.model.predict(X)

# ==================================================
# DATASET
# ==================================================

X, y = load_breast_cancer(
    return_X_y=True
)

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==================================================
# STORAGE
# ==================================================

accs = []
precs = []
recs = []
f1s = []

growth_events = []
final_neurons = []

print("START")

# ==================================================
# 30 RUNS
# ==================================================

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

    gfnn = GFNN()

    gfnn.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    pred = gfnn.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    prec = precision_score(
        y_test,
        pred,
        average="binary"
    )

    rec = recall_score(
        y_test,
        pred,
        average="binary"
    )

    f1 = f1_score(
        y_test,
        pred,
        average="binary"
    )

    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)

    growth_events.append(
        gfnn.growth_events
    )

    final_neurons.append(
        gfnn.final_neurons
    )

# ==================================================
# RESULTS
# ==================================================

print()
print("=" * 60)
print("BREAST CANCER RESULTS")
print("=" * 60)

print(
    "Runs =",
    len(accs)
)

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

print(
    "Average Growth Events =",
    np.mean(growth_events)
)

print(
    "Average Final Neurons =",
    np.mean(final_neurons)
)

print("END")