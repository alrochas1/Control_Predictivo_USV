import numpy as np
from models.ackermann_model import ackermann_model

def calculate_ideal_controls(state, ref_state, max_steering=np.pi/4, max_speed=1.0):

    x, y, theta = state
    x_ref, y_ref, theta_ref = ref_state
    
    # Error entre la posición actual y la deseada
    error_x = x_ref - x
    error_y = y_ref - y
    error_theta = theta_ref - theta

    # Distancia al objetivo
    distance = np.sqrt(error_x**2 + error_y**2)
    v = np.clip(distance, 0, max_speed)  # Velocidad proporcional a la distancia

    # Ángulo de dirección ideal (hacia el objetivo)
    desired_heading = np.arctan2(error_y, error_x)
    phi = desired_heading - theta
    phi = np.clip(phi, -max_steering, max_steering)  # Limitar el ángulo de dirección

    return v, phi

def calculate_ideal_controls_with_initial_adjustment(state, ref_state, max_steering=np.pi/4, max_speed=1.0, adjust=True):

    x, y, theta = state
    x_ref, y_ref, theta_ref = ref_state
    
    # Error entre la posición actual y la deseada
    error_x = x_ref - x
    error_y = y_ref - y
    distance = np.sqrt(error_x**2 + error_y**2)
    
    # Ángulo hacia el punto de referencia
    desired_heading = np.arctan2(error_y, error_x)
    heading_error = desired_heading - theta
    
    # Si estamos ajustando la orientación inicial
    if adjust and abs(heading_error) > 0.1:
        v = 0.2  # Velocidad reducida durante el ajuste inicial
        phi = np.clip(heading_error, -max_steering, max_steering)
    else:
        # Seguimiento normal de la trayectoria
        v = np.clip(distance, 0, max_speed)  # Velocidad proporcional a la distancia
        phi = np.clip(desired_heading - theta, -max_steering, max_steering)  # Ángulo hacia el objetivo
    
    return v, phi

# Parámetros
dt = 0.1
T = 20

# Generación de trayectoria deseada
time = np.linspace(0, T, int(T/dt))
x_ref = time
y_ref = np.sin(time / 5)
theta_ref = np.arctan(np.gradient(y_ref, x_ref))
trajectory = np.vstack((x_ref, y_ref, theta_ref)).T

# Cálculo del ángulo inicial
x_start, y_start = trajectory[0, 0], trajectory[0, 1]
x_next, y_next = trajectory[1, 0], trajectory[1, 1]
initial_theta = np.arctan2(y_next - y_start, x_next - x_start)

# Inicialización de estados y datos
state = np.array([x_start, y_start, initial_theta])  # Estado inicial [x, y, theta]
data_X = []  # Entradas: estado actual + referencia
data_Y = []  # Salidas ideales: [v, phi]

# Simulación para generar datos
for i in range(len(trajectory) - 1):
    ref_state = trajectory[i + 1]
    v, phi = calculate_ideal_controls(state, ref_state)
    
    # Almacenar datos
    data_X.append(np.hstack((state, ref_state)))
    data_Y.append([v, phi])
    
    # Actualizar el estado
    state = ackermann_model(state, [v, phi], dt)

data_X = np.array(data_X)
data_Y = np.array(data_Y)

# Visualización de datos generados
import matplotlib.pyplot as plt
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trayectoria Deseada", linestyle="--")
plt.plot(data_X[:, 0], data_X[:, 1], label="Estados Generados", linestyle="-")
plt.legend()
plt.show()
