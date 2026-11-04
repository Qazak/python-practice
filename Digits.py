import numpy as np

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# GFNN
# =====================================================

class GFNN:

    def __init__(
        self,
        initial_neurons=32,
        growth_step=16,
        max_neurons=256,
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
                activation="relu",
                solver="adam",
                max_iter=50,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
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

            if self.model is None:

                self.model = model
                self.final_neurons = neurons

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

# =====================================================
# LOAD MNIST
# =====================================================

print("Loading MNIST...")

X, y = load_digits(return_X_y=True)

X = X.astype(np.float32)

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# =====================================================
# STORAGE
# =====================================================

accs = []
precs = []
recs = []
f1s = []

growth_events = []
final_neurons = []

# =====================================================
# RUNS
# =====================================================

RUNS = 10

print("START")

for seed in range(RUNS):

    print("\n" + "=" * 60)
    print(f"RUN {seed + 1}/{RUNS}")
    print("=" * 60)

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=seed
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            stratify=y_train_full,
            random_state=seed
        )
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
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        y_test,
        pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0
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

    print(
        f"Accuracy = {acc:.4f}"
    )

    print(
        f"F1 = {f1:.4f}"
    )

    print(
        f"Final Neurons = {gfnn.final_neurons}"
    )

# =====================================================
# RESULTS
# ================================
