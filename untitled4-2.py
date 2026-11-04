import numpy as np

from scipy.stats import ttest_rel

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ====================================================
# GFNN
# ====================================================

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

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(X_val)

            acc = accuracy_score(
                y_val,
                pred
            )

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

# ====================================================
# CONFIDENCE INTERVAL
# ====================================================

def ci95(values):

    values = np.asarray(values)

    mean = values.mean()

    std = values.std(ddof=1)

    margin = (
        1.96
        * std
        / np.sqrt(len(values))
    )

    return (
        mean - margin,
        mean + margin
    )

# ====================================================
# DATA
# ====================================================

X, y = load_breast_cancer(
    return_X_y=True
)

X = StandardScaler().fit_transform(X)

# ====================================================
# STORAGE
# ====================================================

baseline_accs = []

gfnn_accs = []

gfnn_precision = []
gfnn_recall = []
gfnn_f1 = []

growth_events = []
final_neurons = []

# ====================================================
# 30 RUNS
# ====================================================

for seed in range(30):

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )

    # =====================================
    # BASELINE
    # =====================================

    baseline = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=seed
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )

    baseline_accs.append(
        accuracy_score(
            y_test,
            baseline_pred
        )
    )

    # =====================================
    # GFNN
    # =====================================

    gfnn = GFNN()

    gfnn.fit(
        X_train,
        y_train,
        X_test,
        y_test
    )

    pred = gfnn.predict(
        X_test
    )

    gfnn_accs.append(
        accuracy_score(
            y_test,
            pred
        )
    )

    gfnn_precision.append(
        precision_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    gfnn_recall.append(
        recall_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    gfnn_f1.append(
        f1_score(
            y_test,
            pred,
            average="weighted"
        )
    )

    growth_events.append(
        gfnn.growth_events
    )

    final_neurons.append(
        gfnn.final_neurons
    )

# ====================================================
# PAIRED T TEST
# ====================================================

t_stat, p_value = ttest_rel(
    gfnn_accs,
    baseline_accs
)

diff = (
    np.array(gfnn_accs)
    -
    np.array(baseline_accs)
)

cohen_d = (
    diff.mean()
    /
    diff.std(ddof=1)
)

# ====================================================
# RESULTS
# ====================================================

print("\n")
print("=" * 60)
print("TABLE 1 - PERFORMANCE")
print("=" * 60)

print("\nBASELINE FFNN")

print(
    "Accuracy:",
    np.mean(baseline_accs),
    "+/-",
    np.std(
        baseline_accs,
        ddof=1
    )
)

print(
    "95% CI:",
    ci95(baseline_accs)
)

print("\nGFNN")

print(
    "Accuracy:",
    np.mean(gfnn_accs),
    "+/-",
    np.std(
        gfnn_accs,
        ddof=1
    )
)

print(
    "Precision:",
    np.mean(gfnn_precision)
)

print(
    "Recall:",
    np.mean(gfnn_recall)
)

print(
    "F1:",
    np.mean(gfnn_f1)
)

print(
    "95% CI:",
    ci95(gfnn_accs)
)

print(
    "Average Growth Events:",
    np.mean(growth_events)
)

print(
    "Average Final Neurons:",
    np.mean(final_neurons)
)

print("\n")
print("=" * 60)
print("TABLE 2 - STATISTICAL SIGNIFICANCE")
print("=" * 60)

print(
    "Mean Difference:",
    diff.mean()
)

print(
    "t-statistic:",
    t_stat
)

print(
    "p-value:",
    p_value
)

print(
    "Cohen's d:",
    cohen_d
)

if p_value < 0.05:

    print(
        "\nResult: Significant (p < 0.05)"
    )

else:

    print(
        "\nResult: Not Significant"
    )

print("\n")
print("=" * 60)

print(
    "\nBASELINE ACCURACIES:\n",
    baseline_accs
)

print(
    "\nGFNN ACCURACIES:\n",
    gfnn_accs
)

