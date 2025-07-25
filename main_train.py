import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.layers import Dense, Input

import config
from tray import aux
from models.ackermann_model import AckermannModel_tf as USV_Model
from controller.control_activation import ControlActivation


# GPU ------------------------
print("GPUs disponibles:", tf.config.experimental.list_physical_devices('GPU')) 
num_threads = 12
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# ----------------------------

dt = config.dt
N = config.N

x_range = config.x_range
y_range = config.y_range


# def custom_activation(x):

# Modelo -----------------------------------------------
controller = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),  # Añade esta línea
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2),                  # Salida: [v, phi]
    ControlActivation()  # Aplicamos la activación
])



# controller.add(tf.keras.layers.Lambda(custom_activation))  # Aplicamos la activación

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

# -----------------------------------------------
def loss_function(predicted, target, v, phi, state):

    error = predicted - target      # e(t) = -[x_ref - x, y_ref - y, theta_ref - theta]
    # mse = tf.reduce_mean(tf.square(tf.square(error)))/100
    distance_error = tf.norm(error, axis=-1)  # Distancia euclidiana a la trayectoria
    
    # Penalización cuadrática para errores pequeños + exponencial para errores grandes
    mse = tf.reduce_mean(
        tf.where(
            distance_error < 1.0,
            tf.square(distance_error),
            10.0 * tf.exp(distance_error)
        )
    ) / 100
    print(f"MSE: {mse}")

    # Penaliza valores de v < 0.1
    v_penalize = tf.maximum(0.1 - v, 0) ** 2  
    v_penalty = tf.reduce_mean(v_penalize)
    print(f"Error velocidad baja:{v_penalty}")

    # Penaliza mala orientacion
    x, y, theta = state[0], state[1], state[2]
    x_d, y_d = target[0, 0], target[0, 1] 
    desired_theta = tf.atan2(y_d - y, x_d - x)
    theta_error = tf.math.mod(desired_theta - theta + np.pi, 2*np.pi) - np.pi
    theta_penalize = tf.square(theta_error)
    theta_penalty = tf.reduce_mean(theta_penalize)
    print(f"Error Mala Orientacion: {theta_penalty}")

    loss = mse + 1 * v_penalty + 0.1*theta_penalty
    print(f"Perdida total {loss} \n")
    return loss


# Bucle de entrenamiento
model = USV_Model(config.L, config.max_steer)
dt = config.dt
N_epochs = 10


for e in range(N_epochs):
    trajectory = aux.random_bezier_trajectory(N, x_range, y_range)

    theta = aux.compute_start(trajectory)*0
    state = tf.Variable([0.0, 0.0, theta], dtype=tf.float32)  # Posición inicial
    # positions = [state.numpy()[:2]]     # Para graficar
    losses = []

    threshold = 0.3  # Umbral de cercanía
    current_point = 0
    max_steps_per_point = 10
    steps_for_point = 0
    max_steps = max_steps_per_point*len(trajectory)  # Seguridad para evitar bucles infinitos
    steps = 0

    while current_point < len(trajectory) - 1 and steps < max_steps:
        x, y, theta = state.numpy()
        x_d, y_d = trajectory[current_point]

        with tf.GradientTape() as tape:
            control_input = tf.convert_to_tensor([[x, y, theta, x_d, y_d]], dtype=tf.float32)
            control_output = controller(control_input)
            # print(f"Salida de la red: {control_output.numpy()}")

            new_state = model.update(state[0], state[1], state[2],
                                    control_output[:, 0],  # velocidad
                                    control_output[:, 1],  # ángulo
                                    dt)
            # print(f"Salida de la planta: {new_state.numpy()}")

            # Calcular pérdida respecto al punto objetivo
            target_pos = tf.convert_to_tensor([x_d, y_d], dtype=tf.float32)
            pred_pos = new_state[:2]
            v = control_output[:, 0]
            phi = control_output[:, 1]
            loss = loss_function(tf.expand_dims(pred_pos, axis=0),
                                tf.expand_dims(target_pos, axis=0),
                                v, phi, state.numpy())
            # print(f"Perdida: {loss.numpy()}")

        # Entrenamiento
        grads = tape.gradient(loss, controller.trainable_variables)
        optimizer.apply_gradients(zip(grads, controller.trainable_variables))

        # Actualizar estado
        state.assign(tf.reshape(new_state, (3,)))
        losses.append(loss)
        steps += 1

        # Comprobar distancia al punto objetivo
        # Comprobación de distancia
        dist = np.linalg.norm([x - x_d, y - y_d])
        if dist < config.threshold or steps_for_point > max_steps_per_point:
            current_point += 1
            steps_for_point = 0
        else:
            steps_for_point += 1

        if current_point%10 == 0 and steps_for_point == 0:
            print(f"\n Epoca {e+1}/{N_epochs} --- {100*current_point/len(trajectory)}%. Error Actual: {loss.numpy()}")
            print(f"Velocidad Actual: {control_output[:, 0]}, Angulo Actual: {control_output[:, 1]}")

    # for j in range(len(trajectory)-1):  # Iteraciones de entrenamiento
    #     with tf.GradientTape() as tape:

    #         x, y, theta = state.numpy()
    #         x_d, y_d = trajectory[j+1]
    #         print(f"Punto actual: {x_d, y_d}")
    #         control_input = tf.convert_to_tensor([[x, y, theta, x_d, y_d]], dtype=tf.float32)
    #         control_output = controller(control_input)
    #         print(f"Salida de la red: {control_output.numpy()}")

    #         # Aplicar control a la planta
    #         new_state = model.update(state[0], state[1], state[2], 
    #                     control_output[:, 0],
    #                     control_output[:, 1], dt)
    #         print(f"Salida de la planta: {new_state.numpy()}")
            
    #         target_point = trajectory[j+1]
    #         loss = loss_function(tf.convert_to_tensor([new_state[:2]], dtype=tf.float32), 
    #                             tf.convert_to_tensor([target_point], dtype=tf.float32),
    #                             tf.convert_to_tensor(control_output[:, 1], dtype=tf.float32))
    #         losses.append(loss)
    #         print(f"Perdida: {loss.numpy()}")
            
        # grads = tape.gradient(loss, controller.trainable_variables)
        # optimizer.apply_gradients(zip(grads, controller.trainable_variables))
        
        # state.assign(tf.reshape(new_state, (3,)))  # Asegura que el shape sea (5,)
        # # state.assign(tf.convert_to_tensor(new_state, dtype=tf.float32))
        # if j % 100 == 0 or j == len(trajectory) - 1:
        #     print(f"Epoca {e+1}/{N_epochs} --- {100*j/len(trajectory)}%")

    print(f"Época {e+1}, Pérdida Promedio: {np.mean(losses)}")



print("Entrenamiento completado.")
controller.save("usv_controller.keras")
print("Modelo Guardado como usv_controller.keras")




