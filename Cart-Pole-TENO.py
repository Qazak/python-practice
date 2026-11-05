import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

# =====================================================
# CART-POLE PARAMETERS
# =====================================================

g = 9.81
mc = 1.0
mp = 0.1
l = 0.5

# =====================================================
# LQR GAINS
# =====================================================

K = np.array([
    -6.32455532,
    -9.15436964,
    -72.68170406,
    -17.63549353
])

# =====================================================
# CARTPOLE DYNAMICS
# =====================================================

def cartpole(state, u):

    x, xdot, theta, thetadot = state

    s = np.sin(theta)
    c = np.cos(theta)

    thetaddot = (
        g*s
        -
        c * (
            u
            + mp*l*thetadot**2*s
        )/(mc+mp)
    ) / (
        l * (
            4/3
            -
            mp*c*c/(mc+mp)
        )
    )

    xddot = (
        u
        +
        mp*l*(
            thetadot**2*s
            -
            thetaddot*c
        )
    )/(mc+mp)

    return np.array([
        xdot,
        xddot,
        thetadot,
        thetaddot
    ])

# =====================================================
# SIMULATION
# =====================================================

dt = 0.01
T = 10

times = np.arange(
    0,
    T+dt,
    dt
)

x_true = np.array([
    0.0,
    0.0,
    0.05,
    0.0
])

true_states = []

for t in times:

    true_states.append(
        x_true.copy()
    )

    u = -float(
        K @ x_true
    )

    u = np.clip(
        u,
        -20,
        20
    )

    sol = solve_ivp(
        lambda tt, xx:
        cartpole(xx, u),
        [0, dt],
        x_true
    )

    x_true = sol.y[:, -1]

true_states = np.array(
    true_states
)

# =====================================================
# MEASUREMENTS
# =====================================================

measurements = np.column_stack([
    true_states[:,0],
    true_states[:,2]
])

measurements += np.random.normal(
    0,
    0.005,
    measurements.shape
)

# =====================================================
# NORMALIZATION
# =====================================================

Xmean = measurements.mean(0)
Xstd = measurements.std(0)

Ymean = true_states.mean(0)
Ystd = true_states.std(0)

Xtrain = (
    measurements - Xmean
)/Xstd

Ytrain = (
    true_states - Ymean
)/Ystd

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

        self.stop_growth = False

        self.W1 = (
            np.random.randn(
                2,
                hidden
            )
            *
            np.sqrt(2/2)
        )

        self.b1 = np.zeros(hidden)

        self.W2 = (
            np.random.randn(
                hidden,
                4
            )
            *
            np.sqrt(
                2/hidden
            )
        )

        self.b2 = np.zeros(4)

        self.loss_history = []

        self.growth_events = 0

    def act(self,x):

        return np.where(
            x > 0,
            x,
            0.01*x
        )

    def act_grad(self,x):

        return np.where(
            x > 0,
            1.0,
            0.01
        )

    def forward(self,X):

        self.Z1 = X @ self.W1 + self.b1

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

        dW2 = self.H.T @ dY

        db2 = np.sum(
            dY,
            axis=0
        )

        dH = dY @ self.W2.T

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

        self.W2 -= (
            self.lr*dW2
        )

        self.b2 -= (
            self.lr*db2
        )

        self.W1 -= (
            self.lr*dW1
        )

        self.b1 -= (
            self.lr*db1
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
            epoch,
            loss,
            observer.hidden
        )

# =====================================================
# PREDICT
# =====================================================

pred_norm = observer.forward(
    Xtrain
)

pred = (
    pred_norm
    * Ystd
    +
    Ymean
)

# =====================================================
# METRICS
# =====================================================

angle_rmse = np.sqrt(
    np.mean(
        (
            pred[:,2]
            -
            true_states[:,2]
        )**2
    )
)

position_rmse = np.sqrt(
    np.mean(
        (
            pred[:,0]
            -
            true_states[:,0]
        )**2
    )
)

print()
print("="*60)
print("TENO CART-POLE")
print("="*60)

print(
    "Angle RMSE =",
    angle_rmse
)

print(
    "Position RMSE =",
    position_rmse
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
# PLOTS
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(
    times,
    true_states[:,2],
    label="True"
)

plt.plot(
    times,
    pred[:,2],
    "--",
    label="TENO"
)

plt.title(
    "Pole Angle"
)

plt.legend()

plt.subplot(122)

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

plt.title(
    "Cart Position"
)

plt.legend()

plt.tight_layout()
plt.show()
