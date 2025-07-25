import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import config
from tray import aux
from models.ackermann_model import AckermannModel_tf as USV_Model
from controller.control_activation import ControlActivation

# Cargar el modelo entrenado
controller = tf.keras.models.load_model("usv_controller.keras",
                custom_objects={"ControlActivation": ControlActivation})


model = USV_Model()
trajectory = aux.random_bezier_trajectory(config.N, config.x_range, config.y_range, False)

theta = aux.compute_start(trajectory)*0
state = tf.Variable([0.0, 0.0, theta], dtype=tf.float32)
dt = config.dt

# Almacenar posiciones para graficar
positions = [state.numpy()[:2]]

current_point = 0
max_steps = 1000
steps = 0

while current_point < len(trajectory) - 1 and steps < max_steps:
    x, y, theta = state.numpy()
    x_d, y_d = trajectory[current_point]

    control_input = tf.convert_to_tensor([[x, y, theta, x_d, y_d]], dtype=tf.float32)
    control_output = controller(control_input)

    new_state = model.update(state[0], state[1], state[2], 
                             control_output[:, 0],  # velocidad
                             control_output[:, 1],  # ángulo
                             dt)
    
    state.assign(tf.reshape(new_state, (3,)))
    positions.append(state.numpy()[:2])

    dist = np.linalg.norm([x - x_d, y - y_d])
    if dist < config.threshold:
        current_point += 1

    steps += 1

# Convertir lista a numpy para graficar
positions = np.array(positions)

# Graficar la trayectoria de referencia y la ejecutada
plt.plot(trajectory[:, 0], trajectory[:, 1], 'g--', label='Trayectoria deseada')
plt.plot(positions[:, 0], positions[:, 1], 'r-', label='Trayectoria ejecutada')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Prueba del modelo entrenado")
plt.show()
