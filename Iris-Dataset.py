import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# Parameter Counter
# =====================================================

def count_mlp_parameters(input_dim, hidden_dim, output_dim):

    return (
        (input_dim * hidden_dim)
        + hidden_dim
        + (hidden_dim * output_dim)
        + output_dim
    )


# =====================================================
# Evaluation Function
# =====================================================

def evaluate_model(name, model,
                   X_train, y_train,
                   X_test, y_test,
                   params=None):

    start = time.time()

    model.fit(X_train, y_train)

    runtime = time.time() - start

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    return {
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 (%)": round(f1 * 100, 2),
        "Parameters": params,
        "Runtime (s)": round(runtime, 4)
    }


# =====================================================
# Data
# =====================================================

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
output_dim = len(np.unique(y))

results = []

# =====================================================
# MLP-5
# =====================================================

mlp5 = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation="tanh",
    max_iter=1000,
    random_state=42
)

results.append(
    evaluate_model(
        "MLP-5",
        mlp5,
        X_train, y_train,
        X_test, y_test,
        count_mlp_parameters(
            input_dim,
            5,
            output_dim
        )
    )
)

# =====================================================
# MLP-20
# =====================================================

mlp20 = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation="tanh",
    max_iter=1000,
    random_state=42
)

results.append(
    evaluate_model(
        "MLP-20",
        mlp20,
        X_train, y_train,
        X_test, y_test,
        count_mlp_parameters(
            input_dim,
            20,
            output_dim
        )
    )
)

# =====================================================
# MLP-50
# =====================================================

mlp50 = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="tanh",
    max_iter=1000,
    random_state=42
)

results.append(
    evaluate_model(
        "MLP-50",
        mlp50,
        X_train, y_train,
        X_test, y_test,
        count_mlp_parameters(
            input_dim,
            50,
            output_dim
        )
    )
)

# =====================================================
# MLP-93
# Match GFNN final size
# =====================================================

mlp93 = MLPClassifier(
    hidden_layer_sizes=(93,),
    activation="tanh",
    max_iter=1000,
    random_state=42
)

results.append(
    evaluate_model(
        "MLP-93",
        mlp93,
        X_train, y_train,
        X_test, y_test,
        count_mlp_parameters(
            input_dim,
            93,
            output_dim
        )
    )
)

# =====================================================
# SVM-RBF
# =====================================================

svm = SVC(
    kernel="rbf",
    gamma="scale",
    C=1.0,
    random_state=42
)

results.append(
    evaluate_model(
        "SVM-RBF",
        svm,
        X_train, y_train,
        X_test, y_test,
        "-"
    )
)

# =====================================================
# Random Forest
# =====================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

results.append(
    evaluate_model(
        "Random Forest",
        rf,
        X_train, y_train,
        X_test, y_test,
        "-"
    )
)

# =====================================================
# GFNN Result
# Enter your measured values
# =====================================================

results.append({
    "Model": "GFNN",
    "Accuracy (%)": 96.67,
    "Precision (%)": 96.97,
    "Recall (%)": 96.67,
    "F1 (%)": 96.66,
    "Parameters": 747,   # adjust if needed
    "Runtime (s)": "-"
})

# =====================================================
# Final Table
# =====================================================

df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("TABLE 4. COMPARATIVE PERFORMANCE ON THE IRIS DATASET")
print("=" * 100)

print(df.to_string(index=False))

print("\n")
print("=" * 100)
