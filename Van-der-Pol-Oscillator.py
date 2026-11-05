import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# =====================================================
# PARAMETERS
# =====================================================

mu = 1.0

# =====================================================
# LINEARIZATION ABOUT ORIGIN
# =====================================================

A = np.array([
    [0, 1],
    [-1, mu]
])

B = np.array([
    [0],
    [1]
])

# =====================================================
# LQR DESIGN
# =====================================================

Q = np.diag([
    20,
    10
])

R = np.array([[1]])

P = solve_continuous_are(
    A,
    B,
    Q,
    R
)

K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain")
print(K)

# =====================================================
# VAN DER POL DYNAMICS
# =====================================================

def vdp(x, u):

    x1, x2 = x

    dx1 = x2

    dx2 = (
        mu*(1 - x1**2)*x2
        - x1
        + u
    )

    return np.array([
        dx1,
        dx2
    ])

# =====================================================
# SETTINGS
# =====================================================

dt = 0.01
Tfinal = 20

times = np.arange(
    0,
    Tfinal + dt,
    dt
)

# =====================================================
# TRUE STATE
# =====================================================

x_true = np.array([
    2.0,
    0.0
])

# =====================================================
# EKF
# =====================================================

xhat = np.array([
    1.8,
    0.1
])

Pekf = np.eye(2)

Qekf = np.diag([
    1e-5,
    1e-5
])

sigma = 0.05

Rekf = np.array([
    [sigma**2]
])

H = np.array([
    [1,0]
])

I = np.eye(2)

# =====================================================
# STORAGE
# =====================================================

true_states = []
estimated_states = []
controls = []

# =====================================================
# LOOP
# =====================================================

for t in times:

    true_states.append(
        x_true.copy()
    )

    estimated_states.append(
        xhat.copy()
    )

    # -----------------------------------------
    # CONTROL
    # -----------------------------------------

    u = -float(
        K @ xhat.reshape(-1,1)
    )

    u = np.clip(
        u,
        -20,
        20
    )

    controls.append(u)

    # -----------------------------------------
    # TRUE PROPAGATION
    # -----------------------------------------

    sol = solve_ivp(
        lambda tt,xx:
        vdp(xx,u),
        [0,dt],
        x_true
    )

    x_true = sol.y[:,-1]

    # -----------------------------------------
    # MEASUREMENT
    # -----------------------------------------

    z = np.array([
        x_true[0]
        + np.random.normal(
            0,
            sigma
        )
    ])

    # -----------------------------------------
    # EKF PREDICTION
    # -----------------------------------------

    f = vdp(
        xhat,
        u
    )

    xpred = xhat + dt*f

    x1 = xhat[0]
    x2 = xhat[1]

    Fk = np.array([
        [
            0,
            1
        ],
        [
            -2*mu*x1*x2 - 1,
            mu*(1-x1**2)
        ]
    ])

    Fk = np.eye(2) + dt*Fk

    Ppred = (
        Fk @ Pekf @ Fk.T
        + Qekf
    )

    # -----------------------------------------
    # UPDATE
    # -----------------------------------------

    innovation = (
        z
        - H @ xpred
    )

    S = (
        H @ Ppred @ H.T
        + Rekf
    )

    Kgain = (
        Ppred
        @ H.T
        @ np.linalg.inv(S)
    )

    xhat = (
        xpred
        + Kgain @ innovation
    )

    Pekf = (
        I
        - Kgain @ H
    ) @ Ppred

# =====================================================
# ARRAYS
# =====================================================

true_states = np.array(
    true_states
)

estimated_states = np.array(
    estimated_states
)

controls = np.array(
    controls
)

# =====================================================
# METRICS
# =====================================================

state_rmse = np.sqrt(
    np.mean(
        (
            true_states
            -
            estimated_states
        )**2
    )
)

settling_time = Tfinal

for i in range(len(times)):

    if np.all(
        np.abs(
            true_states[i:]
        ) < 0.05
    ):
        settling_time = times[i]
        break

control_cost = np.sum(
    controls**2
)*dt

print()
print("="*60)
print("VAN DER POL EKF + LQR")
print("="*60)

print("State RMSE   =", state_rmse)
print("SettlingTime =", settling_time)
print("Control Cost =", control_cost)

# =====================================================
# PLOTS
# =====================================================

plt.figure(figsize=(12,8))

plt.subplot(221)
plt.plot(
    times,
    true_states[:,0],
    label="True"
)
plt.plot(
    times,
    estimated_states[:,0],
    '--',
    label="EKF"
)
plt.title("State x1")
plt.legend()

plt.subplot(222)
plt.plot(
    times,
    true_states[:,1],
    label="True"
)
plt.plot(
    times,
    estimated_states[:,1],
    '--',
    label="EKF"
)
plt.title("State x2")
plt.legend()

plt.subplot(223)
plt.plot(
    true_states[:,0],
    true_states[:,1]
)
plt.title("Phase Portrait")

plt.subplot(224)
plt.plot(
    times,
    controls
)
plt.title("Control Input")

plt.tight_layout()
plt.show()
