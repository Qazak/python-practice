import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================
# CSTR PARAMETERS
# =====================================================

F = 100.0
V = 100.0

CAf = 1.0
Tf = 350.0

k0 = 5e7
E = 12000

Rgas = 1.987

dH = -5.0e4

rho = 1000.0
Cp = 0.239

UA = 5.0e4

Tc = 300.0

# =====================================================
# CSTR MODEL
# =====================================================

def cstr(x):

    CA, T = x

    rate = (
        k0
        * np.exp(-E/(Rgas*T))
        * CA
    )

    dCA = (
        (F/V)*(CAf - CA)
        - rate
    )

    dT = (
        (F/V)*(Tf - T)
        +
        (-dH/(rho*Cp))*rate
        +
        (UA/(V*rho*Cp))
        *
        (Tc - T)
    )

    return np.array([dCA, dT])

# =====================================================
# UKF PARAMETERS
# =====================================================

n = 2

alpha = 0.1
beta = 2.0
kappa = 0

lam = alpha**2 * (n + kappa) - n

wm = np.zeros(2*n + 1)
wc = np.zeros(2*n + 1)

wm[0] = lam/(n + lam)
wc[0] = wm[0] + (1 - alpha**2 + beta)

for i in range(1, 2*n + 1):
    wm[i] = 1/(2*(n + lam))
    wc[i] = wm[i]

# =====================================================
# INITIAL CONDITIONS
# =====================================================

dt = 0.05

times = np.arange(0, 10 + dt, dt)

x_true = np.array([
    1.0,
    350.0
])

xhat = np.array([
    0.9,
    345.0
])

P = np.eye(2)

Q = np.diag([
    1e-5,
    1e-2
])

sigma_CA = 0.01
sigma_T = 0.5

Rk = np.diag([
    sigma_CA**2,
    sigma_T**2
])

# =====================================================
# STORAGE
# =====================================================

true_states = []
estimated_states = []

# =====================================================
# MAIN LOOP
# =====================================================

for t in times:

    true_states.append(
        x_true.copy()
    )

    estimated_states.append(
        xhat.copy()
    )

    # -----------------------------------------
    # TRUE PLANT
    # -----------------------------------------

    sol = solve_ivp(
        lambda tt,xx: cstr(xx),
        [0, dt],
        x_true,
        method="BDF"
    )

    x_true = sol.y[:, -1]

    # -----------------------------------------
    # MEASUREMENT
    # -----------------------------------------

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

    # -----------------------------------------
    # SIGMA POINTS
    # -----------------------------------------

    Psqrt = np.linalg.cholesky(
        (n + lam) * P
    )

    sigma_pts = np.zeros(
        (2*n + 1, n)
    )

    sigma_pts[0] = xhat

    for i in range(n):

        sigma_pts[i+1] = (
            xhat + Psqrt[:, i]
        )

        sigma_pts[i+1+n] = (
            xhat - Psqrt[:, i]
        )

    # -----------------------------------------
    # PREDICT
    # -----------------------------------------

    sigma_pred = np.zeros_like(
        sigma_pts
    )

    for i in range(2*n + 1):

        dx = cstr(
            sigma_pts[i]
        )

        sigma_pred[i] = (
            sigma_pts[i]
            + dt*dx
        )

    xpred = np.sum(
        wm[:,None]
        * sigma_pred,
        axis=0
    )

    Ppred = Q.copy()

    for i in range(2*n + 1):

        d = (
            sigma_pred[i]
            - xpred
        )

        Ppred += (
            wc[i]
            * np.outer(d,d)
        )

    # -----------------------------------------
    # MEASUREMENT PREDICTION
    # -----------------------------------------

    z_sigma = sigma_pred.copy()

    zpred = np.sum(
        wm[:,None]
        * z_sigma,
        axis=0
    )

    S = Rk.copy()

    Pxz = np.zeros((2,2))

    for i in range(2*n + 1):

        dz = (
            z_sigma[i]
            - zpred
        )

        dx = (
            sigma_pred[i]
            - xpred
        )

        S += (
            wc[i]
            * np.outer(dz,dz)
        )

        Pxz += (
            wc[i]
            * np.outer(dx,dz)
        )

    # -----------------------------------------
    # UPDATE
    # -----------------------------------------

    Kgain = (
        Pxz
        @ np.linalg.inv(S)
    )

    xhat = (
        xpred
        + Kgain
        @ (z - zpred)
    )

    P = (
        Ppred
        - Kgain
        @ S
        @ Kgain.T
    )

# =====================================================
# ARRAYS
# =====================================================

true_states = np.array(
    true_states
)

estimated_states = np.array(
    estimated_states
)

# =====================================================
# METRICS
# =====================================================

ca_rmse = np.sqrt(
    np.mean(
        (
            true_states[:,0]
            -
            estimated_states[:,0]
        )**2
    )
)

temp_rmse = np.sqrt(
    np.mean(
        (
            true_states[:,1]
            -
            estimated_states[:,1]
        )**2
    )
)

print()
print("="*60)
print("NUMPY UKF CSTR RESULTS")
print("="*60)

print("CA RMSE =", ca_rmse)
print("T RMSE  =", temp_rmse)

# =====================================================
# PLOTS
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    times,
    true_states[:,0],
    label="True"
)

plt.plot(
    times,
    estimated_states[:,0],
    "--",
    label="UKF"
)

plt.title("Concentration CA")
plt.legend()

