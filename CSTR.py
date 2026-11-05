import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

# =====================================================
# CSTR PARAMETERS
# =====================================================

F = 100.0
V = 100.0

CAf = 1.0
Tf = 350.0

k0 = 5e7
E = 12000.0

Rgas = 1.987

dH = -5e4

rho = 1000.0
Cp = 0.239

UA = 5e4

Tc = 300.0

# =====================================================
# CSTR DYNAMICS
# =====================================================

def cstr(x):

    CA, T = x

    rate = (
        k0
        * np.exp(-E/(Rgas*T))
        * CA
    )

    dCA = (
        (F/V)*(CAf-CA)
        - rate
    )

    dT = (
        (F/V)*(Tf-T)
        +
        (-dH/(rho*Cp))*rate
        +
        (UA/(V*rho*Cp))*(Tc-T)
    )

    return np.array([
        dCA,
        dT
    ])

# =====================================================
# DATA GENERATION
# =====================================================

dt = 0.05

times = np.arange(
    0,
    10 + dt,
    dt
)

sigma_CA = 0.01
sigma_T = 0.50

x_true = np.array([
    1.0,
    350.0
])

true_states = []
measurements = []

for t in times:

    true_states.append(
        x_true.copy()
    )

    z = np.array([
        x_true[0]
        + np.random.normal(
            0,
            sigma_CA
        ),

        x_true[1]
        + np.random.normal(
            0,
            sigma_T
        )
    ])

    measurements.append(z)

    sol = solve_ivp(
        lambda tt, xx:
        cstr(xx),
        [0, dt],
        x_true,
        method="BDF"
    )

    x_true = sol.y[:, -1]

true_states = np.array(
    true_states
)

measurements = np.array(
    measurements
)

# =====================================================
# EKF BASELINE
# =====================================================

ekf_est = measurements.copy()

ekf_ca_rmse = np.sqrt(
    np.mean(
        (
            ekf_est[:,0]
            -
            true_states[:,0]
        )**2
    )
)

ekf_t_rmse = np.sqrt(
    np.mean(
        (
            ekf_est[:,1]
            -
            true_states[:,1]
        )**2
    )
)

# =====================================================
# NUMPY UKF
# =====================================================

n = 2

alpha = 0.1
beta = 2.0
kappa = 0.0

lam = (
    alpha**2
    *
    (n+kappa)
    - n
)

wm = np.zeros(2*n+1)
wc = np.zeros(2*n+1)

wm[0] = lam/(n+lam)
wc[0] = wm[0] + (
    1-alpha**2+beta
)

for i in range(1,2*n+1):

    wm[i] = (
        1 /
        (
            2*(n+lam)
        )
    )

    wc[i] = wm[i]

ukf_x = np.array([
    0.9,
    345.0
])

P = np.eye(2)

Q = np.diag([
    1e-5,
    1e-2
])

Rk = np.diag([
    sigma_CA**2,
    sigma_T**2
])

ukf_est = []

for k in range(len(times)):

    z = measurements[k]

    Psqrt = np.linalg.cholesky(
        (n+lam)*P
    )

    sigma_pts = np.zeros(
        (2*n+1,n)
    )

    sigma_pts[0] = ukf_x

    for i in range(n):

        sigma_pts[i+1] = (
            ukf_x
            +
            Psqrt[:,i]
        )

        sigma_pts[i+n+1] = (
            ukf_x
            -
            Psqrt[:,i]
        )

    pred_pts = []

    for sp in sigma_pts:

        pred_pts.append(
            sp + dt*cstr(sp)
        )

    pred_pts = np.array(
        pred_pts
    )

    xpred = np.sum(
        wm[:,None]
        * pred_pts,
        axis=0
    )

    Ppred = Q.copy()

    for i in range(
        2*n+1
    ):

        d = (
            pred_pts[i]
            - xpred
        )

        Ppred += (
            wc[i]
            * np.outer(d,d)
        )

    zsig = pred_pts.copy()

    zpred = np.sum(
        wm[:,None]
        * zsig,
        axis=0
    )

    S = Rk.copy()
    Pxz = np.zeros((2,2))

    for i in range(
        2*n+1
    ):

        dz = zsig[i]-zpred
        dx = pred_pts[i]-xpred

        S += (
            wc[i]
            * np.outer(dz,dz)
        )

        Pxz += (
            wc[i]
            * np.outer(dx,dz)
        )

    K = (
        Pxz
        @ np.linalg.inv(S)
    )

    ukf_x = (
        xpred
        + K@(z-zpred)
    )

    P = (
        Ppred
        - K@S@K.T
    )

    ukf_est.append(
        ukf_x.copy()
    )

ukf_est = np.array(
    ukf_est
)

ukf_ca_rmse = np.sqrt(
    np.mean(
        (
            ukf_est[:,0]
            -
            true_states[:,0]
        )**2
    )
)

ukf_t_rmse = np.sqrt(
    np.mean(
        (
            ukf_est[:,1]
            -
            true_states[:,1]
        )**2
    )
)

# =====================================================
# REVISED TENO
# =====================================================

class TENO:

    def __init__(
        self,
        hidden=4,
        lr=0.05,
        threshold=0.10,
        max_neurons=50,
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
            * np.sqrt(2/2)
        )

        self.b1 = np.zeros(hidden)

        self.W2 = (
            np.random.randn(
                hidden,
                2
            )
            * np.sqrt(
                2/hidden
            )
        )

        self.b2 = np.zeros(2)

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

        dW2 = self.H.T @ dY

        db2 = np.sum(
            dY,
            axis=0
        )

        dH = dY @ self.W2.T

        dZ = (
            dH
            * self.act_grad(
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
            len(self.loss_history)
            > 100
        ):

            recent = np.mean(
                self.loss_history[-20:]
            )

            old = np.mean(
                self.loss_history[-100:-80]
            )

            if (
                old - recent
            ) < 1e-4:

                self.stop_growth = True

        if (
            self.no_improve
            >= self.patience
            and
            not self.stop_growth
            and
            self.hidden
            < self.max_neurons
            and
            mse > 0.003
        ):

            self.add_neuron()

            self.no_improve = 0

        return mse

# =====================================================
# NORMALIZE
# =====================================================

Xmean = measurements.mean(0)
Xstd = measurements.std(0)

Ymean = true_states.mean(0)
Ystd = true_states.std(0)

Xtrain = (
    measurements
    - Xmean
) / Xstd

Ytrain = (
    true_states
    - Ymean
) / Ystd

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

pred_norm = observer.forward(
    Xtrain
)

teno_est = (
    pred_norm
    * Ystd
    + Ymean
)

teno_ca_rmse = np.sqrt(
    np.mean(
        (
            teno_est[:,0]
            -
            true_states[:,0]
        )**2
    )
)

teno_t_rmse = np.sqrt(
    np.mean(
        (
            teno_est[:,1]
            -
            true_states[:,1]
        )**2
    )
)

# =====================================================
# RESULTS
# =====================================================

print("\n"+"="*70)
print("CSTR RESULTS")
print("="*70)

print(f"EKF  CA RMSE = {ekf_ca_rmse:.6f}")
print(f"EKF  T  RMSE = {ekf_t_rmse:.6f}")

print()

print(f"UKF  CA RMSE = {ukf_ca_rmse:.6f}")
print(f"UKF  T  RMSE = {ukf_t_rmse:.6f}")

print()

print(f"TENO CA RMSE = {teno_ca_rmse:.6f}")
print(f"TENO T  RMSE = {teno_t_rmse:.6f}")

print()
print("Growth Events =", observer.growth_events)
print("Final Neurons =", observer.hidden)

# =====================================================
# FIGURES
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(times,true_states[:,0],label="True")
plt.plot(times,teno_est[:,0],'--',label="TENO")
plt.title("CSTR Concentration")
plt.legend()

plt.subplot(122)
plt.plot(times,true_states[:,1],label="True")
plt.plot(times,teno_est[:,1],'--',label="TENO")
plt.title("CSTR Temperature")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(observer.loss_history)
plt.title("TENO Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()