import numpy as np

from tensorflow.keras.datasets import fashion_mnist

from sklearn.model_selection import train_test_split
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
        initial_neurons=64,
        growth_step=32,
        max_neurons=512,
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
                max_iter=30,
                early_stopping=True,
                random_state=0
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_val)

            acc = accuracy_score(y_val, pred)

            if self.model is None:

                self.model = model
                self.final_neurons = neurons

            if acc > best_acc:

                best_acc = acc
                self.model = model
                self.final_neurons = neurons

            if (1.0 - acc) <= self.threshold:
                break

            neurons += self.growth_step
            self.growth_events += 1

    def predict(self, X):

        return self.model.predict(X)

# =====================================================
# LOAD FASHION-MNIST
# =====================================================

(X_train_full, y_train_full), (X_test, y_test) = \
    fashion_mnist.load_data()

X_train_full = X_train_full.reshape(
    len(X_train_full),
    784
).astype("float32")

X_test = X_test.reshape(
    len(X_test),
    784
).astype("float32")

X_train_full /= 255.0
X_test /= 255.0

# =====================================================
# STORAGE
# =====================================================

accs = []
precs = []
recs = []
f1s = []

growth_events = []
final_neurons = []

RUNS = 10

for seed in range(RUNS):

    print(f"\nRUN {seed+1}/{RUNS}")

    X_train, X_val, y_train, y_val = \
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
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

    pred = gfnn.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    prec = precision_score(
        y_test,
        pred,
        average="macro"
    )

    rec = recall_score(
        y_test,
        pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        pred,
        average="macro"
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
        f"Accuracy={acc:.4f} "
        f"F1={f1:.4f}"
    )

print("\n")
print("="*60)
print("FASHION-MNIST RESULTS")
print("="*60)

print("Accuracy Mean =", np.mean(accs))
print("Accuracy Std =", np.std(accs))
print("Precision Mean =", np.mean(precs))
print("Recall Mean =", np.mean(recs))
print("F1 Mean =", np.mean(f1s))
print("Average Growth Events =", np.mean(growth_events))
print("Average Final Neurons =", np.mean(final_neurons))
