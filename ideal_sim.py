import numpy as np
import matplotlib.pyplot as plt
from models.ackermann_model import ackermann_model

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

# Simulación del modelo Ackermann
state = np.array([0, 0, 0])  # Estado inicial [x, y, theta]
v = 1.0  # Velocidad constante (m/s)
phi = 0.0  # Ángulo inicial (rad)
states = [state]

for i in range(steps - 1):
    # Ajustar la orientación para seguir la curva
    phi = np.arctan2(y_ref[i + 1] - state[1], x_ref[i + 1] - state[0]) - state[2]
    state = ackermann_model(state, [v, phi], dt)
    states.append(state)

states = np.array(states)


# Visualización
plt.plot(x_ref, y_ref, label='Trayectoria Deseada', linestyle='--')
plt.plot(states[:, 0], states[:, 1], label='Trayectoria Seguida')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación Ideal de Seguimiento')
plt.show()
