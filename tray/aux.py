import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

import config


def compute_start(tray):
    x_start, y_start = tray[0, 0], tray[0, 1]
    x_next, y_next = tray[1, 0], tray[1, 1]
    initial_theta = np.arctan2(y_next - y_start, x_next - x_start)
    return initial_theta

# def get_start_state(tray):
#     x_start, y_start = tray[0, 0], tray[0, 1]
#     x_next, y_next = tray[1, 0], tray[1, 1]
#     initial_theta = np.arctan2(y_next - y_start, x_next - x_start)
#     return initial_theta


def bezier_curve(points, dt):

    num_points = int(100*1/dt)

    n = len(points) - 1                # Grado de la curva
    t = np.linspace(0, 1, num_points)  # Parámetro t
    curve = np.zeros((num_points, 2))  # Inicializar la curva

    for i in range(n + 1):
        binom = comb(n, i)
        term = binom * (1 - t)**(n - i) * t**i
        curve += term[:, np.newaxis] * points[i]

    return curve


def random_bezier_trajectory(num, x_range, y_range, show=True):

    first_point = np.array([[0, 0]])
    last_point = np.array([[x_range[1], np.random.uniform(y_range[0], y_range[1])]])
    intermediate_points = np.column_stack((
        np.random.uniform(x_range[0], x_range[1], num - 2),
        np.random.uniform(y_range[0], y_range[1], num - 2)
    ))
    
    control_points = np.vstack((first_point, intermediate_points, last_point))
    
    trajectory = bezier_curve(control_points, config.dt)

    if show:
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trayectoria de Bézier")
        plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label="Puntos de control")
        plt.plot(control_points[:, 0], control_points[:, 1], '--', color='gray', alpha=0.5, label="Línea de control")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.title("Trayectoria Aleatoria con Curva de Bézier")
        # plt.show()
    
    return trajectory




# Para generar trayectorias

# # Generar Puntos Aleatorios
# def generate_random_points(num_points, x_range, y_range):
#     if num_points < 2:
#         raise ValueError("El número de puntos debe ser al menos 2 para definir el inicio y el final.")

#     x = np.sort(np.random.uniform(x_range[0], x_range[1], num_points - 2))
#     y = np.random.uniform(y_range[0], y_range[1], num_points - 2)

#     x = np.insert(x, 0, 0)
#     y = np.insert(y, 0, 0)
#     x = np.append(x, x_range[1])
#     y = np.append(y, np.random.uniform(y_range[0], y_range[1]))

#     return x, y


# # Interpolación Cúbica
# from scipy.interpolate import CubicSpline

# def cubic_interpolation(x_points, y_points, num_samples):
#     spline = CubicSpline(x_points, y_points)
#     x_new = np.linspace(x_points[0], x_points[-1], num_samples)
#     y_new = spline(x_new)
#     return x_new, y_new

# # Generar la Trayectoria
# def generate_random_trajectory(num_points, x_range, y_range, num_samples):
#     x_points, y_points = generate_random_points(num_points, x_range, y_range)
#     x, y = cubic_interpolation(x_points, y_points, num_samples)
#     theta = np.arctan2(np.gradient(y), np.gradient(x))  # Orientación basada en las derivadas
#     return np.vstack((x, y, theta)).T


