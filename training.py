import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import clear_output
# from sklearn.model_selection import train_test_split

from tray import aux
from models.ackermann_model import ackermann_model as ackermann_model
from controller.pd_controller import pd_controller

# Parámetros del sistema
N = 5
N_t = 400
dt = 0.1  # Paso de tiempo
epochs = 100  # Número de épocas de entrenamiento
batch_size = 32  # Tamaño de los lotes

# Generar datos de entrenamiento
def generate_training_data(num_trajectories=10, num_points_per_trajectory=200):

    data_X = []  # Entradas: estado actual + objetivo
    data_Y = []  # Salidas: [nuevo estado]

    for _ in range(num_trajectories):
        trajectory = aux.generate_random_trajectory(num_points=N, 
                                                   x_range=(0, 100), 
                                                   y_range=(-1, 1), 
                                                   num_samples=N*N_t)
        initial_theta = aux.get_start_state(trajectory)
        state = np.array([trajectory[0, 0], trajectory[0, 1], initial_theta])  # Estado inicial

        for i in range(len(trajectory) - 1):
            target_state = trajectory[i + 1]

            # Guardar entrada y salida
            input_data = np.hstack((state, target_state))  # [x, y, theta, x_ref, y_ref, theta_ref]
            output_data = target_state  # Estado deseado después de aplicar control
            data_X.append(input_data)
            data_Y.append(output_data)

            # Actualizar el estado (simulación)
            state = target_state

    return np.array(data_X), np.array(data_Y)



# Definir la función personalizada de pérdida (MSE entre el nuevo estado y el estado objetivo)
dt = tf.constant(0.1, dtype=tf.float32)

def custom_loss(y_true, y_pred):

    v_pred = y_pred[:, 0]
    phi_pred = y_pred[:, 1]

    x_ref = y_true[:, 0]
    y_ref = y_true[:, 1]
    theta_ref = y_true[:, 2]

    # Simular el nuevo estado usando el modelo Ackermann (versión tensorial)
    l = 2.5
    x_dot = v_pred * tf.cos(theta_ref)
    y_dot = v_pred * tf.sin(theta_ref)
    theta_dot = v_pred * tf.tan(phi_pred) / l

    x_pred = x_ref + x_dot * dt
    y_pred = y_ref + y_dot * dt
    theta_pred = theta_ref + theta_dot * dt

    mse = tf.reduce_mean(tf.square(y_true - tf.stack([x_pred, y_pred, theta_pred], axis=1)))
    return mse


# Generar datos
data_X, data_Y = generate_training_data()

# # Dividir datos en conjuntos de entrenamiento y validación
# # X_train, X_val, Y_train, Y_val = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)
# data_Y = 2

# Definir la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Entrada: [x, y, theta, x_ref, y_ref, theta_ref]
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Salida: [v, phi]
])

model.compile(optimizer='adam', loss=custom_loss)


# Callback para graficar trayectorias deseadas vs. trayectorias predichas
class TrajectoryPlotCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        # Seleccionar una de las trayectorias usadas en esta iteración
        idx = np.random.randint(0, len(self.trajectories))  # Escoger una trayectoria aleatoria
        trajectory = self.trajectories[idx]
        initial_state = trajectory[0]
        predicted_states = [initial_state]

        # Simular la trayectoria usando la red neuronal
        for i in range(len(trajectory) - 1):
            target_state = trajectory[i + 1]
            input_data = np.hstack((predicted_states[-1], target_state))
            action = model.predict(input_data.reshape(1, -1), verbose=0)[0]
            new_state = ackermann_model(predicted_states[-1], action, dt)
            predicted_states.append(new_state)

        predicted_states = np.array(predicted_states)

        # Graficar la trayectoria deseada vs. la predicha
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trayectoria Deseada", linestyle="--")
        plt.plot(predicted_states[:, 0], predicted_states[:, 1], label="Trayectoria Seguida", color="orange")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Trayectorias (Época {epoch + 1}) - Pérdida: {logs['loss']:.4f}")
        plt.legend()
        plt.grid()
        plt.pause(0.1)
        plt.close()



# Entrenar la red
history = model.fit(
    data_X, data_Y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[TrajectoryPlotCallback()]
)

# Guardar los pesos de la red
model.save("controlador_nn.keras")

# Guardar el historial de entrenamiento para análisis posterior
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Entrenamiento completado. Pesos guardados en 'controlador_nn.keras'.")

