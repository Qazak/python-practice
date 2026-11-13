import time
import warnings
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================================
# 1. GFNN HELPER FUNCTIONS (WITH PROPER OPTIMIZER LIFTING)
# ==========================================================
def count_params(model):
    return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

def compute_synaptic_divergence(model, X):
    Z = X @ model.coefs_[0] + model.intercepts_[0]
    A = np.tanh(Z)
    D_j = np.mean(np.abs(A - np.mean(A, axis=0, keepdims=True)), axis=0)
    return D_j, float(np.max(D_j))

def expand_mlp_function_preserving(old_model, X_train, y_train, K_new, spot_idx,
                                   epochs_per_stage, seed):
    rng = np.random.RandomState(seed)
    W_in, W_out = old_model.coefs_[0], old_model.coefs_[1]
    b_in, b_out = old_model.intercepts_[0], old_model.intercepts_[1]

    # Localized perturbation around selected Topological Spot
    u_new = W_in[:, [spot_idx]] + rng.normal(0.0, 0.15, size=(W_in.shape[0], K_new))
    b_new = b_in[[spot_idx]] + rng.normal(0.0, 0.15, size=(K_new,))
    # Lemma 1: Strictly zero outgoing weights (exact functional invariance)
    v_new = np.zeros((K_new, W_out.shape[1]))

    new_H = W_in.shape[1] + K_new
    new_model = MLPClassifier(
        hidden_layer_sizes=(new_H,),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.005,
        max_iter=1,
        warm_start=True,
        random_state=seed
    )
    # Initialize internal sklearn structures with full class set, then inject lifted weights
    new_model.fit(X_train, y_train)
    new_model.coefs_ = [np.hstack([W_in, u_new]), np.vstack([W_out, v_new])]
    new_model.intercepts_ = [np.concatenate([b_in, b_new]), b_out.copy()]
    new_model.n_iter_ = 0
    new_model.max_iter = epochs_per_stage
    new_model.fit(X_train, y_train)
    return new_model

def train_gfnn_variant(X_train, y_train, mode="full", H0=8, tau=0.84, max_cycles=5,
                       epochs_per_stage=350, seed=0):
    t0 = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(H0,),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.005,
        max_iter=epochs_per_stage,
        random_state=seed
    )
    model.fit(X_train, y_train)
    D_j, D_curr = compute_synaptic_divergence(model, X_train)
    D_init = D_curr
    growth_events = 0

    if mode != "fixed":
        F_prev, F_curr = 1, 1
        for r in range(1, max_cycles + 1):
            should_grow = (D_curr > tau) if mode in ("gfnn_a", "full") else (r <= 2)
            if not should_grow:
                break

            K_r = 1 if mode == "gfnn_a" else F_curr
            spot_idx = int(np.argmax(D_j)) if mode in ("gfnn_a", "full") else (r % len(D_j))

            model = expand_mlp_function_preserving(
                model, X_train, y_train, K_r, spot_idx, epochs_per_stage, seed + r * 100
            )
            growth_events += 1
            D_j, D_curr = compute_synaptic_divergence(model, X_train)

            if mode in ("gfnn_b", "full"):
                F_prev, F_curr = F_curr, F_prev + F_curr

    train_time_ms = (time.perf_counter() - t0) * 1000.0
    return model, train_time_ms, growth_events, D_init, D_curr

def measure_single_inference_us(model, X_test, n_repeats=200):
    x_single = X_test[:1]
    for _ in range(10):
        model.predict(x_single)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(x_single)
    return ((time.perf_counter() - t0) / n_repeats) * 1e6

# ==========================================================
# 2. MAIN BENCHMARK LOOP (4 DATASETS x 30 RUNS)
# ==========================================================
datasets = {
    "Iris": load_iris(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
    "Digits": load_digits(return_X_y=True),
}

for ds_name, (X, y) in datasets.items():
    print("\n" + "=" * 125)
    print(f"DATASET: {ds_name}")
    print("=" * 125)
    multiclass = len(np.unique(y)) > 2
    avg_mode = "macro" if multiclass else "binary"

    methods = ["MLP-8", "MLP-16", "MLP-32", "MLP-64", "GFNN-A", "GFNN-B", "Proposed GFNN"]
    res = {m: {"acc": [], "f1": [], "params": [], "neurons": [],
               "train_ms": [], "infer_us": [], "growth": [], "d_drop": []} for m in methods}

    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # 1) Fixed-Topology MLP Baselines
        for h in [8, 16, 32, 64]:
            name = f"MLP-{h}"
            t0 = time.perf_counter()
            clf = MLPClassifier(
                hidden_layer_sizes=(h,),
                activation="tanh",
                max_iter=2000,
                random_state=seed
            )
            clf.fit(X_tr, y_tr)
            t_ms = (time.perf_counter() - t0) * 1000.0
            y_pred = clf.predict(X_te)

            res[name]["acc"].append(accuracy_score(y_te, y_pred) * 100.0)
            res[name]["f1"].append(f1_score(y_te, y_pred, average=avg_mode))
            res[name]["params"].append(count_params(clf))
            res[name]["neurons"].append(h)
            res[name]["train_ms"].append(t_ms)
            res[name]["infer_us"].append(measure_single_inference_us(clf, X_te))
            res[name]["growth"].append(0)
            res[name]["d_drop"].append(0.0)

        # 2) GFNN Ablation Variants & Full Proposed Model
        for mode, name in [("gfnn_a", "GFNN-A"), ("gfnn_b", "GFNN-B"), ("full", "Proposed GFNN")]:
            clf, t_ms, g_cnt, d0, dr = train_gfnn_variant(X_tr, y_tr, mode=mode, seed=seed)
            y_pred = clf.predict(X_te)
            drop_pct = ((d0 - dr) / d0 * 100.0) if d0 > 0 else 0.0

            res[name]["acc"].append(accuracy_score(y_te, y_pred) * 100.0)
            res[name]["f1"].append(f1_score(y_te, y_pred, average=avg_mode))
            res[name]["params"].append(count_params(clf))
            res[name]["neurons"].append(clf.coefs_[0].shape[1])
            res[name]["train_ms"].append(t_ms)
            res[name]["infer_us"].append(measure_single_inference_us(clf, X_te))
            res[name]["growth"].append(g_cnt)
            res[name]["d_drop"].append(drop_pct)

    gfnn_acc = np.array(res["Proposed GFNN"]["acc"])
    print(f"{'Model':14s} | {'Acc (% ± SD)':14s} | {'F1':6s} | {'Neurons':7s} | {'Params':7s} | "
          f"{'Train(ms)':9s} | {'Infer(us)':9s} | {'Growth':6s} | {'D_drop%':7s} | {'p-val (Wilc / t)'}")
    print("-" * 125)
    for m in methods:
        acc_arr = np.array(res[m]["acc"])
        if m == "Proposed GFNN":
            p_str = "Reference"
        else:
            diff = gfnn_acc - acc_arr
            if np.allclose(diff, 0):
                p_w, p_t = 1.0, 1.0
            else:
                _, p_w = wilcoxon(gfnn_acc, acc_arr)
                _, p_t = ttest_rel(gfnn_acc, acc_arr)
            p_str = f"{p_w:.4f} / {p_t:.4f}"

        print(f"{m:14s} | {np.mean(acc_arr):5.2f} ± {np.std(acc_arr):4.2f} | "
              f"{np.mean(res[m]['f1']):.4f} | {np.mean(res[m]['neurons']):7.2f} | "
              f"{np.mean(res[m]['params']):7.1f} | {np.mean(res[m]['train_ms']):9.2f} | "
              f"{np.mean(res[m]['infer_us']):9.2f} | {np.mean(res[m]['growth']):6.2f} | "
              f"{np.mean(res[m]['d_drop']):6.1f}% | {p_str}")

print("\nFINISHED")