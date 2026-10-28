import random
patients = [
    {"id": 1, "age": 65, "tumor_volume": 0.80, "ctdna": 0.50, "toxicity": 0.20, "survival": 24},
    {"id": 2, "age": 58, "tumor_volume": 0.60, "ctdna": 0.40, "toxicity": 0.30, "survival": 36},
    {"id": 3, "age": 71, "tumor_volume": 0.90, "ctdna": 0.70, "toxicity": 0.40, "survival": 18},
    {"id": 4, "age": 55, "tumor_volume": 0.40, "ctdna": 0.20, "toxicity": 0.10, "survival": 48},
    {"id": 5, "age": 62, "tumor_volume": 0.75, "ctdna": 0.55, "toxicity": 0.25, "survival": 30}
]

# =====================================================
# Реконструкция состояния (Neural Operator surrogate)
# =====================================================

def reconstruct(patient):

    x0 = (
        0.5 * patient["tumor_volume"]
        + 0.3 * patient["ctdna"]
        - 0.2 * patient["toxicity"]
    )

    theta = patient["age"] / 100.0

    return x0, theta


# =====================================================
# Улучшенное управление
# =====================================================

def optimal_control(x):

    target = 0.10

    k = 3.0

    u = k * (x - target)

    if u < 0:
        u = 0

    if u > 1:
        u = 1

    return u


# =====================================================
# Динамика опухоли
# =====================================================

def tumor_dynamics(x, u, theta):

    growth = theta * x * (1.0 - x)

    treatment = 1.5 * u * x

    return growth - treatment


# =====================================================
# Симуляция пациента
# =====================================================

def simulate_patient(patient):

    x0, theta = reconstruct(patient)

    x = max(0.1, abs(x0))

    dt = 0.1
    steps = 50

    trajectory = []

    for _ in range(steps):

        u = optimal_control(x)

        dx = tumor_dynamics(x, u, theta)

        x = x + dt * dx

        if x < 0:
            x = 0

        trajectory.append(x)

    final_tumor = trajectory[-1]

    predicted_survival = 60 * (1 - final_tumor)

    error = abs(predicted_survival - patient["survival"])

    return {
        "id": patient["id"],
        "initial": round(x0, 4),
        "final": round(final_tumor, 4),
        "predicted_survival": round(predicted_survival, 2),
        "actual_survival": patient["survival"],
        "error": round(error, 2),
        "success": final_tumor < x0
    }


# =====================================================
# Валидация всей когорты
# =====================================================

results = []

for patient in patients:

    result = simulate_patient(patient)

    results.append(result)

# =====================================================
# Вывод результатов
# =====================================================

print("VALIDATION OF 5 PATIENTS")
print()

success_count = 0
total_error = 0

for r in results:

    if r["success"]:
        success_count += 1

    total_error += r["error"]

    print("Patient", r["id"])
    print(" Initial tumor =", r["initial"])
    print(" Final tumor   =", r["final"])
    print(" Predicted survival =", r["predicted_survival"])
    print(" Actual survival    =", r["actual_survival"])
    print(" Absolute error     =", r["error"])

    if r["success"]:
        print(" Result: SUCCESS")
    else:
        print(" Result: FAILURE")

    print()

mae = total_error / len(results)

print("--------------------------------------")
print("Patients:", len(results))
print("Successful controls:", success_count)
print("Control success rate:",
      round(100 * success_count / len(results), 1), "%")
print("Mean Absolute Error:", round(mae, 2), "months")
print("--------------------------------------")

# =====================================================
# Итоговое заключение
# =====================================================

if success_count >= 4:

    print()
    print("Conclusion:")
    print("The proposed digital twin and optimal control")
    print("framework achieved tumor reduction in the")
    print("majority of virtual patients.")