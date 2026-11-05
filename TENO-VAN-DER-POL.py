import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

# =====================================================
# VAN DER POL PARAMETERS
# =====================================================

mu = 1.0

# =====================================================
# DYNAMICS
# =====================================================

def vdp(x, u):

    x1 = x[0]
    x2 = x[1]

    dx1 = x2

    dx2 = (
        mu*(1-x1**2)*x2
        - x1
        + u
    )

    return np.array([
        dx1,
        dx2
    ])

# =====================================================
# LQR GAIN
# =====================================================

K = np.array([
    3.5826,
    5.2440
])

# =====================================================
# SIMULATION
# =====================================================

dt = 0.01

Tfinal = 20

times = np.arange(
    0,
    Tfinal + dt,
    dt
)

x_true = np.array([
    2.0,
    0.0
])

true_states = []

for t in times:

    true_states.append(
        x_true.copy()
    )

    u = -float(K @ x_true)

    u = np.clip(
        u,
        -20,
        20
    )

    sol = solve_ivp(
        lambda tt,xx:
        vdp(xx,u),
        [0,dt],
        x_true
    )

    x_true = sol.y[:,-1]

true_states = np.array(
    true_states
)

# =====================================================
# MEASUREMENTS
# =====================================================

measurements = (
    true_states[:,0]
    +
    np.random.normal(
        0,
        0.05,
        len(times)
    )
).reshape(-1,1)

# =====================================================
# NORMALIZATION
# =====================================================

Xmean = measurements.mean(0)
Xstd = measurements.std(0)

Ymean = true_states.mean(0)
Ystd = true_states.std(0)

Xtrain = (
    measurements - Xmean
) / Xstd

Ytrain = (
    true_states - Ymean
) / Ystd

# =====================================================
# TENO
# =====================================================

class TENO:

    def __init__(
        self,
        hidden=4,
        lr=0.05,
        threshold=0.10,
        max_neurons=40,
        patience=20
    ):

        self.hidden = hidden
        self.lr = lr

        self.threshold = threshold

        self.max_neurons = max_neurons

        self.patience = patience

        self.best_loss = np.inf
        self.no_improve = 0

        self.growth_events = 0

        self.W1 = (
            np.random.randn(
                1,
                hidden
            )
            * np.sqrt(2)
        )

        self.b1 = np.zeros(hidden)

        self.W2 = (
            np.random.randn(
                hidden,
                2
            )
            *
            np.sqrt(
                2/hidden
            )
        )

        self.b2 = np.zeros(2)

        self.loss_history = []

    def act(self,x):

        return np.where(
            x > 0,
            x,
            0.01*x
        )

    def act_grad(self,x):

        return np.where(
            x > 0,
            1,
            0.01
        )

    def forward(self,X):

        self.Z1 = (
            X @ self.W1
            + self.b1
        )

        self.H = self.act(
            self.Z1
        )

        return (
            self.H @ self.W2
            + self.b2
        )

    def add_neuron(self):

        self.hidden += 1

        self.growth_events += 1

        self.W1 = np.hstack([
            self.W1,
            np.random.randn(
                self.W1.shape[0],
                1
            )*0.01
        ])

        self.b1 = np.append(
            self.b1,
            0
        )

        self.W2 = np.vstack([
            self.W2,
            np.random.randn(
                1,
                self.W2.shape[1]
            )*0.01
        ])

    def train_step(
        self,
        X,
        Y
    ):

        pred = self.forward(X)

        err = pred - Y

        mse = np.mean(
            err**2
        )

        self.loss_history.append(
            mse
        )

        if mse < (
            self.best_loss
            - 1e-4
        ):

            self.best_loss = mse
            self.no_improve = 0

        else:

            self.no_improve += 1

        dY = (
            2*err
        ) / len(X)

        dW2 = (
            self.H.T @ dY
        )

        db2 = np.sum(
            dY,
            axis=0
        )

        dH = (
            dY @ self.W2.T
        )

        dZ = (
            dH
            *
            self.act_grad(
                self.Z1
            )
        )

        dW1 = X.T @ dZ

        db1 = np.sum(
            dZ,
            axis=0
        )

        self.W1 -= (
            self.lr*dW1
        )

        self.b1 -= (
            self.lr*db1
        )

        self.W2 -= (
            self.lr*dW2
        )

        self.b2 -= (
            self.lr*db2
        )

        if (
            self.no_improve
            >= self.patience
            and
            self.hidden
            <
            self.max_neurons
            and
            mse > 0.003
        ):

            self.add_neuron()

            self.no_improve = 0

        return mse

# =====================================================
# TRAIN
# =====================================================

observer = TENO()

for epoch in range(1000):

    loss = observer.train_step(
        Xtrain,
        Ytrain
    )

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch}",
            f"Loss {loss:.6f}",
            f"Hidden {observer.hidden}"
        )

# =====================================================
# PREDICTION
# =====================================================

pred_norm = observer.forward(
    Xtrain
)

pred = (
    pred_norm
    *
    Ystd
    +
    Ymean
)

# =====================================================
# METRICS
# =====================================================

state_rmse = np.sqrt(
    np.mean(
        (
            pred
            -
            true_states
        )**2
    )
)

print()
print("="*60)
print("TENO VAN DER POL")
print("="*60)

print(
    "State RMSE =",
    state_rmse
)

print(
    "Growth Events =",
    observer.growth_events
)

print(
    "Final Neurons =",
    observer.hidden
)

# =====================================================
# FIGURES
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(
    times,
    true_states[:,0],
    label="True"
)

plt.plot(
    times,
    pred[:,0],
    "--",
    label="TENO"
)

plt.title("State x1")
plt.legend()

plt.subplot(122)

plt.plot(
    times,
    true_states[:,1],
    label="True"
)

plt.plot(
    times,
    pred[:,1],
    "--",
    label="TENO"
)

plt.title("State x2")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure()

plt.plot(
    observer.loss_history
)

plt.title(
    "TENO Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.grid(True)

plt.show()
