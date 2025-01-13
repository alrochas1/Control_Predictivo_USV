import numpy as np

from models.ackermann_model import ackermann_model
from controller.pd_controller import pd_controller

# Generar Puntos Aleatorios
def generate_random_points(num_points, x_range, y_range):
    x = np.sort(np.random.uniform(x_range[0], x_range[1], num_points))  # Ordenar para evitar retrocesos
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    return x, y

# Interpolación Cúbica
from scipy.interpolate import CubicSpline

def cubic_interpolation(x_points, y_points, num_samples):
    spline = CubicSpline(x_points, y_points)
    x_new = np.linspace(x_points[0], x_points[-1], num_samples)
    y_new = spline(x_new)
    return x_new, y_new

# Generar la Trayectoria
def generate_random_trajectory(num_points, x_range, y_range, num_samples):
    x_points, y_points = generate_random_points(num_points, x_range, y_range)
    x, y = cubic_interpolation(x_points, y_points, num_samples)
    theta = np.arctan2(np.gradient(y), np.gradient(x))  # Orientación basada en las derivadas
    return np.vstack((x, y, theta)).T


# Visualizar Trayectorias Generadas
import matplotlib.pyplot as plt

# Generar varias trayectorias aleatorias
N = 15
trajectories = [generate_random_trajectory(num_points=N, 
                                           x_range=(0, 10), 
                                           y_range=(-5, 5), 
                                           num_samples=N*50) for _ in range(2)]

# Visualización
plt.figure(figsize=(10, 6))
for trajectory in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trayectoria")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trayectorias Generadas con Interpolación Cúbica")
plt.legend()
plt.show()


# Uso en el Entrenamiento
# data_X = []  # Entradas: [x, y, theta, x_ref, y_ref, theta_ref]
# data_Y = []  # Salidas: [v, phi]

# for trajectory in trajectories:
#     state = np.array([0, 0, 0])  # Estado inicial
#     for i in range(len(trajectory) - 1):
#         target_state = trajectory[i + 1]
#         phi = pd_controller(state, target_state)  # Controlador PD
#         v = 1.0  # Velocidad constante
#         state = ackermann_model(state, [v, phi], dt)
        
#         # Almacena datos para entrenamiento
#         input_data = np.hstack((state, target_state))
#         output_data = [v, phi]
#         data_X.append(input_data)
#         data_Y.append(output_data)

