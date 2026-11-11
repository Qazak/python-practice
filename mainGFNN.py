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


# =========================================================
# Synaptic Divergence
# =========================================================

def compute_synaptic_divergence(model, X):

    hidden_input = np.dot(X, model.coefs_[0]) + model.intercepts_[0]

    hidden_output = np.tanh(hidden_input)

    mean_activation = np.mean(hidden_output, axis=0)

    divergence = np.mean(
        np.abs(hidden_output - mean_activation),
        axis=0
    )

    return divergence


# =========================================================
# GFNN
# =========================================================

def train_gfnn(
        X_train,
        y_train,
        X_test,
        y_test,
        tau=0.70,
        max_growth_rounds=20,
        initial_neurons=5):

    neurons = initial_neurons

    fib_prev = 1
    fib_curr = 1

    history = []

    print("=" * 60)
    print("Growing Feedforward Neural Network")
    print("=" * 60)

    final_model = None

    for growth_round in range(max_growth_rounds):

        current_neurons = neurons

        model = MLPClassifier(
            hidden_layer_sizes=(current_neurons,),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)

        div_values = compute_synaptic_divergence(
            model,
            X_train
        )

        d_max = np.max(div_values)
        d_mean = np.mean(div_values)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"\nRound {growth_round + 1}")
        print(f"Neurons        : {current_neurons}")
        print(f"Dmax           : {d_max:.4f}")
        print(f"Dmean          : {d_mean:.4f}")
        print(f"Accuracy       : {acc:.4f}")

        history.append({
            "round": growth_round + 1,
            "neurons": current_neurons,
            "dmax": d_max,
            "dmean": d_mean,
            "accuracy": acc
        })

        final_model = model

        # ==========================================
        # STOPPING CONDITION
        # ==========================================
        if d_max <= tau:

            print("\nGrowth terminated.")
            print(
                f"Dmax ({d_max:.4f}) <= tau ({tau:.4f})"
            )

            break

        # ==========================================
        # FIBONACCI GROWTH
        # ==========================================

        if growth_round == 0:
            neurons += 1

        elif growth_round == 1:
            neurons += 1

        else:
            new_neurons = fib_prev + fib_curr

            fib_prev, fib_curr = (
                fib_curr,
                new_neurons
            )

            neurons += new_neurons

    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    final_predictions = final_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        final_predictions
    )

    precision = precision_score(
        y_test,
        final_predictions,
        average="macro"
    )

    recall = recall_score(
        y_test,
        final_predictions,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        final_predictions,
        average="macro"
    )

    results = {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        # Correct final architecture
        "final_neurons": history[-1]["neurons"],

        "growth_rounds": len(history),

        "final_dmax": history[-1]["dmax"],

        "history": history
    }

    return final_model, results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

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

    model, results = train_gfnn(
        X_train,
        y_train,
        X_test,
        y_test,

        # stop earlier
        tau=0.70,

        max_growth_rounds=20,

        initial_neurons=5
    )

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Accuracy       : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision      : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1-score       : "
        f"{results['f1']:.4f}"
    )

    print(
        f"Final Neurons  : "
        f"{results['final_neurons']}"
    )

    print(
        f"Growth Rounds  : "
        f"{results['growth_rounds']}"
    )

    print(
        f"Final Dmax     : "
        f"{results['final_dmax']:.4f}"
    )

    print("\nGrowth History")
    print("-" * 60)

    for h in results["history"]:

        print(
            f"Round {h['round']:2d} | "
            f"Neurons={h['neurons']:3d} | "
            f"Dmax={h['dmax']:.4f} | "
            f"Acc={h['accuracy']:.4f}"
        )