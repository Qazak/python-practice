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

k0 = 7.2e8
E = 12000.0

R = 1.987

dH = -5.0e4

rho = 1000.0
Cp = 0.239

UA = 5.0e4

Tc = 300.0

# =====================================================
# CSTR MODEL
# =====================================================

def cstr_dynamics(x):

    CA = x[0]
    T = x[1]

    rate = k0 * np.exp(-E / (R * T)) * CA

    dCA = (F / V) * (CAf - CA) - rate

    dT = (
        (F / V) * (Tf - T)
        + (-dH / (rho * Cp)) * rate
        + (UA / (V * rho * Cp)) * (Tc - T)
    )

    return np.array([dCA, dT])


# =====================================================
# TRUE SIMULATION
# =====================================================

dt = 0.05
Tfinal = 10.0

times = np.arange(0, Tfinal + dt, dt)

x0 = np.array([1.0, 350.0])

true_states = np.zeros((len(times), 2))
true_states[0] = x0

for k in range(len(times) - 1):

    sol = solve_ivp(
        lambda t, x: cstr_dynamics(x),
        [0, dt],
        true_states[k],
        method="BDF"
    )

    true_states[k + 1] = sol.y[:, -1]

# =====================================================
# NOISY MEASUREMENTS
# =====================================================

np.random.seed(42)

sigma_CA = 0.01
sigma_T = 0.5

measurements = true_states.copy()

measurements[:, 0] += np.random.normal(
    0,
    sigma_CA,
    len(times)
)

measurements[:, 1] += np.random.normal(
    0,
    sigma_T,
    len(times)
)

# =====================================================
# EKF SETTINGS
# =====================================================

xhat = np.array([0.8, 330.0])

P = np.eye(2)

Q = np.diag([1e-5, 1e-2])

Rk = np.diag([
    sigma_CA**2,
    sigma_T**2
])

I = np.eye(2)

estimated = np.zeros_like(true_states)

# =====================================================
# EKF LOOP
# =====================================================

for k in range(len(times)):

    estimated[k] = xhat

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    f = cstr_dynamics(xhat)

    xpred = xhat + dt * f

    CA = xhat[0]
    T = xhat[1]

    exp_term = np.exp(-E / (R * T))

    dr_dCA = k0 * exp_term

    dr_dT = (
        k0
        * CA
        * exp_term
        * E
        / (R * T**2)
    )

    A = np.array([
        [
            -(F / V) - dr_dCA,
            -dr_dT
        ],
        [
            (-dH / (rho * Cp)) * dr_dCA,
            -(F / V)
            + (-dH / (rho * Cp)) * dr_dT
            - UA / (V * rho * Cp)
        ]
    ])

    Fk = I + dt * A

    Ppred = Fk @ P @ Fk.T + Q

    # ---------------------------------------------
    # Measurement Update
    # ---------------------------------------------

    H = np.eye(2)

    z = measurements[k]

    y = z - xpred

    S = H @ Ppred @ H.T + Rk

    K = Ppred @ H.T @ np.linalg.inv(S)

    xhat = xpred + K @ y

    P = (I - K @ H) @ Ppred

# =====================================================
# RMSE
# =====================================================

rmse_CA = np.sqrt(
    np.mean(
        (true_states[:, 0] - estimated[:, 0]) ** 2
    )
)

rmse_T = np.sqrt(
    np.mean(
        (true_states[:, 1] - estimated[:, 1]) ** 2
    )
)

print("=" * 60)
print("CSTR EKF RESULTS")
print("=" * 60)

print("CA RMSE =", rmse_CA)
print("T RMSE =", rmse_T)

# =====================================================
# PLOTS
# =====================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    times,
    true_states[:, 0],
    label="True CA",
    linewidth=2
)

plt.plot(
    times,
    estimated[:, 0],
    "--",
    label="EKF CA"
)

plt.scatter(
    times,
    measurements[:, 0],
    s=10,
    alpha=0.4,
    label="Measured"
)

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    times,
    true_states[:, 1],
    label="True T",
    linewidth=2
)

plt.plot(
    times,
    estimated[:, 1],
    "--",
    label="EKF T"
)

plt.scatter(
    times,
    measurements[:, 1],
    s=10,
    alpha=0.4,
    label="Measured"
)

plt.xlabel("Time")
plt.ylabel("Temperature (K)")
plt.legend()

plt.tight_layout()
plt.show()
mae_CA = np.mean(
    np.abs(true_states[:,0] - estimated[:,0])
)

mae_T = np.mean(
    np.abs(true_states[:,1] - estimated[:,1])
)

print("CA RMSE =", rmse_CA)
print("CA MAE  =", mae_CA)

print("T RMSE =", rmse_T)
print("T MAE  =", mae_T)
mae_CA = np.mean(
    np.abs(true_states[:,0] - estimated[:,0])
)

mae_T = np.mean(
    np.abs(true_states[:,1] - estimated[:,1])
)

max_CA_error = np.max(
    np.abs(true_states[:,0] - estimated[:,0])
)

max_T_error = np.max(
    np.abs(true_states[:,1] - estimated[:,1])
)

print("\nCSTR EKF RESULTS")
print("="*50)

print("CA RMSE =", rmse_CA)
print("CA MAE  =", mae_CA)
print("CA MAX  =", max_CA_error)

print()

print("T RMSE =", rmse_T)
print("T MAE  =", mae_T)
print("T MAX  =", max_T_error)