import numpy as np
import matplotlib.pyplot as plt

from models.ackermann_model import ackermann_model
from controller.pd_controller import pd_controller


# Parámetros de simulación
dt = 0.1  # Paso de tiempo (s)
T = 20    # Tiempo total (s)
steps = int(T / dt)

# Trayectoria deseada (curva sinusoidal)
time = np.linspace(0, T, steps)
x_ref = time
y_ref = np.sin(time / 5)
theta_ref = np.arctan(np.gradient(y_ref, x_ref))  # Aproximación de orientación
trajectory = np.vstack((x_ref, y_ref, theta_ref)).T

# Cálculo del ángulo inicial ------------------------------
x_start, y_start = trajectory[0, 0], trajectory[0, 1]
x_next, y_next = trajectory[1, 0], trajectory[1, 1]
initial_theta = np.arctan2(y_next - y_start, x_next - x_start)

# Simulación del modelo Ackermann
state = np.array([x_start, y_start, initial_theta])  # Estado inicial [x, y, theta]
v = 1.0  # Velocidad constante (m/s)
states = [state]

for i in range(steps - 1):
    target_state = trajectory[i + 1]
    phi = pd_controller(state, target_state)
    state = ackermann_model(state, [v, phi], dt)
    states.append(state)

states = np.array(states)

# Visualización
plt.plot(x_ref, y_ref, label='Trayectoria Deseada', linestyle='--')
plt.plot(states[:, 0], states[:, 1], label='Trayectoria Seguida (PD Control)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación con Controlador PD')
plt.show()
