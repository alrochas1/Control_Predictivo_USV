import random
import numpy as np
import matplotlib.pyplot as plt
from models.ackermann_model import ackermann_model

def add_noise(value, noise_level):
    return value + np.random.normal(0, noise_level)


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

# Simulación con ruido
noisy_states = [np.array([0, 0, 0])]  # Estado inicial con ruido

for i in range(steps - 1):
    phi = np.arctan2(y_ref[i + 1] - noisy_states[-1][1], x_ref[i + 1] - noisy_states[-1][0]) - noisy_states[-1][2]
    v_noisy = add_noise(v, 0.1)  # Ruido en velocidad
    phi_noisy = add_noise(phi, np.radians(10))  # Ruido en ángulo
    state = ackermann_model(noisy_states[-1], [v_noisy, phi_noisy], dt)
    noisy_states.append(state)

noisy_states = np.array(noisy_states)

# Visualización con ruido
plt.plot(x_ref, y_ref, label='Trayectoria Deseada', linestyle='--')
plt.plot(noisy_states[:, 0], noisy_states[:, 1], label='Trayectoria con Ruido')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación con Ruido e Imperfecciones')
plt.show()
