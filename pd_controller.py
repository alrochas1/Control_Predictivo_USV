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

N = 5
N_t = 400
dt = 1/N/N_t*50
trajectory = training.sin_tray(dt, 50)
# trajectory = aux.generate_random_trajectory(num_points=N, 
                                        #    x_range=(0, 100), 
                                        #    y_range=(-1, 1), 
                                        #    num_samples=N*N_t)


initial_theta = aux.get_start_state(trajectory)
state = np.array([trajectory[0, 0], trajectory[0, 1], initial_theta])  # Estado inicial [x, y, theta]
v = 1.0
states = [state]

for i in range(len(trajectory) - 1):
    target_state = trajectory[i + 1]
    phi = pd_controller(state, target_state, 20, 0.001)
    state = model(state, [v, phi], dt)
    states.append(state)

states = np.array(states)

mse = calculate_error(states, trajectory)
print(f"Error cuadrático medio (MSE): {mse:.4f}")

# Visualización
plt.figure(1)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria Deseada', linestyle='--')
plt.plot(states[:, 0], states[:, 1], label='Trayectoria Seguida (PD Control)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación con Controlador PD')
# plt.figure(2)
# t = np.linspace(0, len(states[:, 2]), len(states[:, 2]))
# plt.plot(t, states[:, 2], label='Giro')
# plt.xlabel('t')
# plt.ylabel('Theta')
# plt.title('Evolución del Giro')
plt.show()


