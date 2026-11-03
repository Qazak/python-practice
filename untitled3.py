import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class GFNN:

    def __init__(
        self,
        initial_neurons=4,
        growth_step=2,
        max_neurons=32,
        threshold=0.05
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

        neurons = self.initial_neurons

        best_acc = -1

        while neurons <= self.max_neurons:

            model = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                activation="relu",
                solver="adam",
                max_iter=2000,
                random_state=0
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_val)

            acc = accuracy_score(y_val, pred)

            discrepancy = 1.0 - acc

            if acc > best_acc:
                best_acc = acc
                self.model = model
                self.final_neurons = neurons

            if discrepancy <= self.threshold:
                break

            neurons += self.growth_step
            self.growth_events += 1

    def predict(self, X):
        return self.model.predict(X)
      
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


X, y = load_breast_cancer(return_X_y=True)

X = StandardScaler().fit_transform(X)

accs = []
precs = []
recs = []
f1s = []

growth_events = []
final_neurons = []

for seed in range(30):

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )

    model = GFNN(
        initial_neurons=4,
        growth_step=2,
        max_neurons=32,
        threshold=0.05
    )

    model.fit(
        X_train,
        y_train,
        X_test,
        y_test
    )

    pred = model.predict(X_test)

    accs.append(
        accuracy_score(y_test, pred)
    )

    precs.append(
        precision_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    recs.append(
        recall_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    f1s.append(
        f1_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    growth_events.append(
        model.growth_events
    )

    final_neurons.append(
        model.final_neurons
    )

print("\nGFNN RESULTS")
print("=" * 40)

print("Accuracy Mean:",
      np.mean(accs))

print("Accuracy Std:",
      np.std(accs, ddof=1))

print("Precision Mean:",
      np.mean(precs))

print("Recall Mean:",
      np.mean(recs))

print("F1 Mean:",
      np.mean(f1s))

print("Average Growth Events:",
      np.mean(growth_events))

print("Average Final Neurons:",
      np.mean(final_neurons))# New Python file
