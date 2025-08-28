import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

import config


def compute_start(tray):
    x_start, y_start = tray[0, 0], tray[0, 1]
    x_next, y_next = tray[1, 0], tray[1, 1]
    initial_theta = np.arctan2(y_next - y_start, x_next - x_start)
    return initial_theta


def bezier_curve(points, dt):

    num_points = int(5/dt)

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


def predefined_trajectory(show=True):
    first_point = np.array([[0, 0]])
    last_point = np.array([[config.tray_x[7], config.tray_y[7]]])
    # Define puntos intermedios predefinidos para la trayectoria
    intermediate_points = np.array([
        [config.tray_x[1], config.tray_y[1]],
        [config.tray_x[2], config.tray_y[2]],
        [config.tray_x[3], config.tray_y[3]],
        [config.tray_x[4], config.tray_y[4]],
        [config.tray_x[5], config.tray_y[5]],
        [config.tray_x[6], config.tray_y[6]],  
    ])

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


# Usado por Gym
def get_reference_trajectory(predefined=False):
    
    # Obtener los puntos [x, y]
    if predefined:
        xy_trajectory = predefined_trajectory(show=False)
    else:
        xy_trajectory = random_bezier_trajectory(num=config.N, x_range=config.x_range, y_range=config.y_range, show=False)

    # Calcula la orientación (yaw) deseada para cada punto de la trayectoria.
    # La orientación se aproxima por la derivada: theta = arctan2(dy, dx)
    dx = np.diff(xy_trajectory[:, 0])
    dy = np.diff(xy_trajectory[:, 1])
    
    # Calcula el yaw para cada segmento
    yaw = np.arctan2(dy, dx)
    # El último punto tiene el mismo yaw que el penúltimo
    yaw = np.append(yaw, yaw[-1])
    
    # Construye la trayectoria completa: [x, y, theta]
    full_trajectory = np.column_stack((xy_trajectory, yaw))
    return full_trajectory


# Calcula la diferencia más corta entre dos ángulos (alpha - beta)
# en el rango [-pi, pi] de forma robusta.
def angle_difference(alpha, beta):
    diff = alpha - beta
    return np.arctan2(np.sin(diff), np.cos(diff))
