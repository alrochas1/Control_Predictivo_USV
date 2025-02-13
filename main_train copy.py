import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input

import config
from tray import aux
# from models.ackermann_model import AckermannModel as model

# usv_model = model()
dt = config.dt

@tf.function
def modelo_ackermann_tf(x, y, theta, v, delta):

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    theta = tf.cast(theta, tf.float32)
    v = tf.cast(v, tf.float32)
    delta = tf.cast(delta, tf.float32)

    x_next = x + v * tf.cos(theta) * dt
    y_next = y + v * tf.sin(theta) * dt
    theta_next = theta + (v / config.L) * tf.tan(delta) * dt
    return x_next, y_next, theta_next


# def generar_datos_bezier(num_trayectorias=1000):
#     X_train = []  # Entrada (x, y, theta, x_d, y_d)
#     Y_train = []  # Salida (v, delta)

#     for _ in range(num_trayectorias):

#         trayectoria = aux.random_bezier_trajectory(num=4, x_range=[0, 10], y_range=[-5, 5])
#         x, y, theta = 0, 0, 0

#         for i in range(len(trayectoria) - 1):
#             # Punto de referencia actual
#             x_d, y_d = trayectoria[i+1]

#             # Calcular error de dirección
#             error_x = x_d - x
#             error_y = y_d - y
#             theta_ref = np.arctan2(error_y, error_x)  # Orientación deseada

#             # Generar v y delta simples (puedes mejorar esto con un controlador PID o RL)
#             v = min(1.0, np.hypot(error_x, error_y))  # Limitar velocidad
#             delta = np.clip(theta_ref - theta, -0.4, 0.4)  # Limitar el giro

#             # Guardar datos
#             X_train.append([x, y, theta, x_d, y_d])
#             Y_train.append([v, delta])

#             # Aplicar modelo de Ackermann y avanzar
#             x, y, theta = usv_model.update(x, y, theta, v, delta)

#     return np.array(X_train), np.array(Y_train)


def generate_batch_trajectories(N_trajectories, N_steps, x_range, y_range):
    X_batch = []
    Y_batch = []

    for _ in range(N_trajectories):
        traj = aux.random_bezier_trajectory(N, x_range, y_range, False)  # Generar una trayectoria Bézier
        X_batch.append(traj[:N_steps])  # Tomamos N_steps puntos

    return np.array(X_batch)


# Crear la red neuronal
model = keras.Sequential([
    Input(shape=(5,)),  # (x, y, theta, x_d, y_d)
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='tanh')
])


@tf.function
def loss_fn(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    x, y, theta = y_true[:, 0], y_true[:, 1], y_true[:, 2]  # Estado actual
    x_d, y_d = y_true[:, 3], y_true[:, 4]  # Estado deseado
    v, delta = y_pred[:, 0], y_pred[:, 1]  # Salidas de la red

    # Simular el modelo de Ackermann
    x_next, y_next, theta_next = modelo_ackermann_tf(x, y, theta, v, delta)

    # Error de seguimiento: distancia entre la posición simulada y la deseada
    error_pos = (x_next - x_d) ** 2 + (y_next - y_d) ** 2

    # Regularización para suavizar la velocidad y el giro
    error_suavidad = 0.1 * (v ** 2 + delta ** 2)

    return tf.reduce_mean(error_pos + error_suavidad)


@tf.function
def train_step(X_traj):
    batch_size = X_traj.shape[0]  # Numero de trayectorias en paralelo
    N_steps = X_traj.shape[1]     # Numero de puntos por trayectoria

    x = tf.zeros((batch_size,))
    y = tf.zeros((batch_size,))
    theta = tf.zeros((batch_size,))

    total_loss = 0.0

    for t in range(N_steps):
        x_d = X_traj[:, t, 0]  # Objetivo x en todas las trayectorias
        y_d = X_traj[:, t, 1]  # Objetivo y en todas las trayectorias

        X_input = tf.stack([x, y, theta, x_d, y_d], axis=1)  # Shape: (batch_size, 5)

        with tf.GradientTape() as tape:
            Y_pred = model(X_input, training=True) 
            v, delta = Y_pred[:, 0], Y_pred[:, 1]

            x_next, y_next, theta_next = modelo_ackermann_tf(x, y, theta, v, delta)

            # Calcular error de seguimiento
            error_pos = (x_next - x_d) ** 2 + (y_next - y_d) ** 2
            error_suavidad = 0.1 * (v ** 2 + delta ** 2)

            loss = tf.reduce_mean(error_pos + error_suavidad)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Actualizar estados del vehículo
        x, y, theta = x_next, y_next, theta_next

        total_loss += loss

    return total_loss / N_steps


# Compilar el modelo (el loss se define más adelante)
optimizer = keras.optimizers.Adam(learning_rate=0.01)

N_samples = 50
N_epoch = 10

N_trajectories = 5
N_steps = int(100*1/dt)
# X_train = []  # (x, y, theta, x_d, y_d)
# Y_train = []  # (v, delta)

N = config.N
x_range = config.x_range
y_range = config.y_range

print("Empezando el entrenamiento ...")
for epoch in range(N_epoch):  # Número de épocas
    losses = []

    for i in range(N_samples):
        X_batch = generate_batch_trajectories(N_trajectories, N_steps, x_range, y_range)
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)  # Convertir a tensor

        loss = train_step(X_batch)  # Entrenamiento en batch
        losses.append(loss.numpy())
        # traj = aux.random_bezier_trajectory(N, x_range, y_range) 
        # x, y, theta = 0, 0, 0  # Estado inicial

        # for point in traj:
        #     x_d, y_d = point  # Punto de referencia

        #     X_input = np.array([[x, y, theta, x_d, y_d]])
        #     with tf.GradientTape() as tape:
        #         Y_pred = model(X_input, training=True)
        #         loss = loss_fn(np.array([[x, y, theta, x_d, y_d]]), Y_pred)

        #     # Aplicar gradientes
        #     grads = tape.gradient(loss, model.trainable_variables)
        #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #     # Guardar pérdidas para análisis
        #     losses.append(loss.numpy())
        #     # print(f"Punto={point}")
            
        print(f"Epoca {epoch+1}: {i}/{N_samples} --- {(i/N_samples/N_epoch):.2f}%")

    print("--------------------\n")
    print(f"Época {epoch+1}, Pérdida Promedio: {np.mean(losses)}")



# Guardar el modelo después del entrenamiento
model.save("modelo_ackermann.h5")
print("Modelo guardado como modelo_ackermann.h5")


# # ENTRENAR
# model.compile(optimizer=optimizer, loss=loss_fn)
# model.fit(X_train, Y_train, epochs=100, batch_size=32)

test_traj = aux.random_bezier_trajectory(N, x_range, y_range)
x_actual, y_actual, theta_actual = 0, 0, 0

plt.figure()
plt.plot(test_traj[:, 0], test_traj[:, 1], label="Trayectoria deseada", color="blue")

for point in test_traj:
    x_deseado, y_deseado = point
    X_input = np.array([[x_actual, y_actual, theta_actual, x_deseado, y_deseado]])
    v_pred, delta_pred = model.predict(X_input)[0]

    x_actual, y_actual, theta_actual = modelo_ackermann_tf(x_actual, y_actual, theta_actual, v_pred, delta_pred)

    plt.scatter(x_actual, y_actual, color="red", s=10)

plt.legend()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Seguimiento de Trayectoria con la Red Neuronal")
plt.show()










