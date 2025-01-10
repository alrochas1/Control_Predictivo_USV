import tensorflow as tf

# Red neuronal para el controlador
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Entrada: estado actual + punto objetivo
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Salida: [v, phi]
])
model.compile(optimizer='adam', loss='mse')


trajectory = generate_sine_wave(amplitude=2, wavelength=5, length=10, points=100)
state = [0, 0, 0]  # x, y, theta

import numpy as np

# Parámetros de entrenamiento
epochs = 100
dt = 0.1  # Paso de tiempo

for epoch in range(epochs):
    state = np.array([0, 0, 0])  # Estado inicial
    total_loss = 0

    for i in range(len(trajectory) - 1):
        target_state = trajectory[i + 1]  # Siguiente punto de la trayectoria
        input_data = np.hstack((state, target_state))  # Entrada para la red
        
        # Predicción de la red neuronal
        action = model.predict(input_data.reshape(1, -1), verbose=0)[0]  # [v, phi]
        
        # Simulación del modelo con la acción predicha
        new_state = ackermann_model(state, action, dt)
        
        # Cálculo del error
        error = target_state - new_state
        loss = np.mean(error**2)
        total_loss += loss
        
        # Entrenamiento de la red
        model.fit(input_data.reshape(1, -1), action.reshape(1, -1), verbose=0)
        
        # Actualización del estado
        state = new_state

    print(f"Epoch {epoch + 1}/{epochs}, Pérdida total: {total_loss:.4f}")


# Simulación final con la red entrenada
state = np.array([0, 0, 0])
nn_states = [state]

for i in range(len(trajectory) - 1):
    target_state = trajectory[i + 1]
    input_data = np.hstack((state, target_state))
    action = model.predict(input_data.reshape(1, -1), verbose=0)[0]
    state = ackermann_model(state, action, dt)
    nn_states.append(state)

nn_states = np.array(nn_states)

# Visualización
import matplotlib.pyplot as plt
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trayectoria Deseada", linestyle='--')
plt.plot(nn_states[:, 0], nn_states[:, 1], label="Trayectoria Seguida (NN)")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Desempeño del Controlador con Red Neuronal")
plt.show()

