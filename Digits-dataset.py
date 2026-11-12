import warnings
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

# =====================================================
# GFNN
# =====================================================

class GFNN:

    def __init__(
        self,
        initial_neurons=8,
        growth_step=2,
        max_neurons=66,
        threshold=0.02,
        patience=3
    ):

        self.initial_neurons = initial_neurons
        self.growth_step = growth_step
        self.max_neurons = max_neurons
        self.threshold = threshold
        self.patience = patience

        self.model = None
        self.best_neurons = initial_neurons

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=0
    ):

        neurons = self.initial_neurons

        best_acc = -1.0
        best_score = -1.0

        patience_counter = 0

        while neurons <= self.max_neurons:

            model = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                max_iter=2000,
                random_state=random_state
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            acc = accuracy_score(
                y_val,
                y_pred
            )

            score = acc - 0.001 * neurons

            print(
                f"Neurons={neurons:2d} "
                f"Acc={acc:.4f} "
                f"Score={score:.4f}"
            )

            improved = False

            if score > best_score + 1e-4:
                improved = True

            elif abs(score - best_score) <= 0.01:
                if neurons < self.best_neurons:
                    improved = True

            if improved:

                best_score = score
                self.model = model
                self.best_neurons = neurons

                patience_counter = 0

            else:

                patience_counter += 1

            if (1.0 - acc) <= self.threshold:
                break

            if patience_counter >= self.patience:
                break

            neurons += self.growth_step

    def predict(self, X):

        return self.model.predict(X)


# =====================================================
# LOAD DIGITS DATASET
# =====================================================

X, y = load_digits(return_X_y=True)

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =====================================================
# STORAGE
# =====================================================

accs = []
precs = []
recs = []
f1s = []
neurons_used = []

print("START")

# =====================================================
# 30 RUNS
# =====================================================

for seed in range(30):

    print("\nRUN", seed + 1)

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
            test_size=0.25,
            stratify=y_train_full,
            random_state=seed
        )
    )

    model = GFNN(
        initial_neurons=8,
        growth_step=2,
        max_neurons=66,
        threshold=0.02,
        patience=3
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=seed
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

    neurons_used.append(
        model.best_neurons
    )

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 60)
print("DIGITS RESULTS")
print("=" * 60)

print("Runs =", len(accs))

print(
    "Accuracy Mean =",
    round(np.mean(accs), 4)
)

print(
    "Accuracy Std =",
    round(np.std(accs), 4)
)

print(
    "Precision Mean =",
    round(np.mean(precs), 4)
)

print(
    "Recall Mean =",
    round(np.mean(recs), 4)
)

print(
    "F1 Mean =",
    round(np.mean(f1s), 4)
)

print(
    "Average Final Neurons =",
    round(np.mean(neurons_used), 2)
)

print(
    "Min Neurons =",
    np.min(neurons_used)
)

print(
    "Max Neurons =",
    np.max(neurons_used)
)

print("=" * 60)
print("END")