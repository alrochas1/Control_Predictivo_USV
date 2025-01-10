import numpy as np
import matplotlib.pyplot as plt

from tray import aux, training 

from models.ackermann_model import ackermann_model_noise as model
from controller.pd_controller import pd_controller
from error import calculate_error


# Parámetros de simulación
dt = 0.1  # Paso de tiempo (s)
T = 20    # Tiempo total (s)

# Simulación del modelo Ackermann
trajectory = training.sin_tray(dt, T)
initial_theta = aux.get_start_state(trajectory)
state = np.array([0, 0, initial_theta])  # Estado inicial [x, y, theta]
v = 1.0
states = [state]

for i in range(int(T/dt) - 1):
    target_state = trajectory[i + 1]
    phi = pd_controller(state, target_state)
    state = model(state, [v, phi], dt)
    states.append(state)

states = np.array(states)

mse = calculate_error(states, trajectory)
print(f"Error cuadrático medio (MSE): {mse:.4f}")

# Visualización
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria Deseada', linestyle='--')
plt.plot(states[:, 0], states[:, 1], label='Trayectoria Seguida (PD Control)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación con Controlador PD')
plt.show()


