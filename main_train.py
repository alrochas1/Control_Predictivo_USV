import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.layers import Dense, Input

import config
from tray import aux
from models.ackermann_model import AckermannModel_tf as USV_Model


# GPU ------------------------
print("GPUs disponibles:", tf.config.experimental.list_physical_devices('GPU')) 
num_threads = 6
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# ----------------------------

dt = config.dt
N = config.N

x_range = config.x_range
y_range = config.y_range

# Modelo -----------------------------------------------
controller = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)  # Salida: [v, phi]
])

def custom_activation(x):
    v = tf.nn.softplus(x[:, 0])  # Garantiza que v >= 0
    phi = x[:, 1]  # phi puede ser cualquier valor, ya está limitado en el modelo
    return tf.stack([v, phi], axis=1)

controller.add(tf.keras.layers.Lambda(custom_activation))  # Aplicamos la activación

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# -----------------------------------------------


def loss_function(predicted, target):
    error = target - predicted  # e(t) = [x_ref - x, y_ref - y, theta_ref - theta]
    return tf.reduce_mean(tf.norm(error, axis=1))


# Bucle de entrenamiento
model = USV_Model(config.L, config.max_steer)
dt = config.dt
N_epochs = 100


for e in range(N_epochs):
    trajectory = aux.random_bezier_trajectory(N, x_range, y_range)
    state = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)  # Posición inicial
    # state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    losses = []

    for j in range(len(trajectory)-1):  # Iteraciones de entrenamiento
        with tf.GradientTape() as tape:

            x, y, theta = state.numpy()
            x_d, y_d = trajectory[j+1]
            # print(f"Punto actual: {x_d, y_d}")
            control_input = tf.convert_to_tensor([[x, y, theta, x_d, y_d]], dtype=tf.float32)
            control_output = controller(control_input)
            # print(f"Salida de la red: {control_output.numpy()}")

            # Aplicar control a la planta
            new_state = model.update(state[0], state[1], state[2], 
                        control_output[:, 0],
                        control_output[:, 1], dt)
            # print(f"Salida de la planta: {new_state.numpy()}")
            
            target_point = trajectory[j]
            loss = loss_function(tf.convert_to_tensor([new_state[:2]], dtype=tf.float32), 
                                tf.convert_to_tensor([target_point], dtype=tf.float32))
            losses.append(loss)
            # print(f"Perdida: {loss.numpy()}")
            
        grads = tape.gradient(loss, controller.trainable_variables)
        optimizer.apply_gradients(zip(grads, controller.trainable_variables))
        
        state.assign(tf.reshape(new_state, (3,)))  # Asegura que el shape sea (5,)
        # state.assign(tf.convert_to_tensor(new_state, dtype=tf.float32))
        if j % 100 == 0 or j == len(trajectory) - 1:
            print(f"Epoca {e+1}/{N_epochs} --- {100*j/len(trajectory)}%")

    print(f"Época {e+1}, Pérdida Promedio: {np.mean(losses)}")



print("Entrenamiento completado.")
controller.save("usv_controller.keras")
print("Modelo Guardado como usv_controller.keras")










