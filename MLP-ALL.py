import warnings
import numpy as np
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits
)
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================================
# DATASETS
# ==========================================================
datasets = {
    "Iris": load_iris(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
    "Digits": load_digits(return_X_y=True)
}

# ==========================================================
# BASELINES
# ==========================================================
baseline_sizes = [8, 16, 32, 64]

# ==========================================================
# MAIN LOOP (LEAKAGE-FREE & COMPACT OUTPUT)
# ==========================================================
for dataset_name, (X, y) in datasets.items():
    multiclass = len(np.unique(y)) > 2

    for hidden_neurons in baseline_sizes:
        accs, precs, recs, f1s = [], [], [], []

        for seed in range(30):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=seed
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = MLPClassifier(
                hidden_layer_sizes=(hidden_neurons,),
                max_iter=2000,
                random_state=seed
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            avg_mode = "macro" if multiclass else "binary"
            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, average=avg_mode))
            recs.append(recall_score(y_test, y_pred, average=avg_mode))
            f1s.append(f1_score(y_test, y_pred, average=avg_mode))

        print(
            f"{dataset_name:13s} | MLP-{hidden_neurons:<2d} | "
            f"Acc: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}% | "
            f"P: {np.mean(precs):.4f} | R: {np.mean(recs):.4f} | F1: {np.mean(f1s):.4f}"
        )

print("\nFINISHED")
