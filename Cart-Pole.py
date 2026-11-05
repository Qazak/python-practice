import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# =====================================================
# PARAMETERS
# =====================================================

g = 9.81

mc = 1.0
mp = 0.1

l = 0.5

# =====================================================
# STANDARD UPRIGHT LINEARIZATION
# =====================================================

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(mp*g)/mc, 0],
    [0, 0, 0, 1],
    [0, 0, ((mc+mp)*g)/(l*mc), 0]
])

B = np.array([
    [0],
    [1/mc],
    [0],
    [-1/(l*mc)]
])

# =====================================================
# LQR
# =====================================================

Q = np.diag([
    20,
    2,
    500,
    50
])

R = np.array([[0.5]])

P = solve_continuous_are(A,B,Q,R)

K = np.linalg.inv(R) @ B.T @ P

print("\nLQR Gain")
print(K)

# =====================================================
# NONLINEAR UPRIGHT CART-POLE
# =====================================================

def cartpole(state,u):

    x,xdot,theta,thetadot = state

    s = np.sin(theta)
    c = np.cos(theta)

    denom = mc + mp*s*s

    thetaddot = (
        g*s
        - c*(u + mp*l*thetadot**2*s)/(mc+mp)
    ) / (
        l*(4.0/3.0 - mp*c*c/(mc+mp))
    )

    xddot = (
        u
        + mp*l*(
            thetadot**2*s
            - thetaddot*c
        )
    ) / (mc+mp)

    return np.array([
        xdot,
        xddot,
        thetadot,
        thetaddot
    ])

# =====================================================
# SETTINGS
# =====================================================

dt = 0.01
T = 10

times = np.arange(
    0,
    T + dt,
    dt
)

# =====================================================
# TRUE STATE
# =====================================================

x_true = np.array([
    0.0,
    0.0,
    0.05,
    0.0
])

# =====================================================
# EKF
# =====================================================

xhat = np.array([
    0.0,
    0.0,
    0.04,
    0.0
])

Pekf = np.eye(4)

Qekf = np.diag([
    1e-5,
    1e-4,
    1e-5,
    1e-4
])

sigma_x = 0.005
sigma_theta = 0.005

Rekf = np.diag([
    sigma_x**2,
    sigma_theta**2
])

H = np.array([
    [1,0,0,0],
    [0,0,1,0]
])

I = np.eye(4)

# =====================================================
# STORAGE
# =====================================================

true_states = []
estimated_states = []
controls = []

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
    # LQR CONTROL
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
        cartpole(xx,u),
        [0,dt],
        x_true,
        method="RK45"
    )

    x_true = sol.y[:,-1]

    # angle wrap

    x_true[2] = np.arctan2(
        np.sin(x_true[2]),
        np.cos(x_true[2])
    )

    # -----------------------------------------
    # MEASUREMENT
    # -----------------------------------------

    z = np.array([
        x_true[0],
        x_true[2]
    ])

    z += np.array([
        np.random.normal(
            0,
            sigma_x
        ),
        np.random.normal(
            0,
            sigma_theta
        )
    ])

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    f = cartpole(
        xhat,
        u
    )

    xpred = xhat + dt*f

    eps = 1e-6

    Fk = np.zeros((4,4))

    for i in range(4):

        dx = np.zeros(4)
        dx[i] = eps

        f1 = cartpole(
            xhat + dx,
            u
        )

        f2 = cartpole(
            xhat - dx,
            u
        )

        Fk[:,i] = (
            f1 - f2
        )/(2*eps)

    Fk = np.eye(4) + dt*Fk

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

angle_rmse = np.sqrt(
    np.mean(
        (
            true_states[:,2]
            -
            estimated_states[:,2]
        )**2
    )
)

position_rmse = np.sqrt(
    np.mean(
        (
            true_states[:,0]
            -
            estimated_states[:,0]
        )**2
    )
)

settling_time = T

for i in range(len(times)):

    if np.all(
        np.abs(
            true_states[i:,2]
        ) < 0.01
    ):

        settling_time = times[i]
        break

Qcost = np.diag([
    20,
    2,
    500,
    20
])

control_cost = 0

for k in range(len(times)):

    x = true_states[k]
    u = controls[k]

    control_cost += (
        x.T @ Qcost @ x
        +
        u*u
    ) * dt

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("="*60)
print("CART-POLE EKF + LQR")
print("="*60)

print("Angle RMSE    =", angle_rmse)
print("Position RMSE =", position_rmse)
print("Settling Time =", settling_time)
print("Control Cost  =", control_cost)

# ==================