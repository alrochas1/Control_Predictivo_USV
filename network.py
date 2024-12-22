import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np

# Generación de datos de entrenamiento
X = []  # Estados actuales + trayectoria deseada
Y = []  # Entradas óptimas [v, phi]

for state, ref in zip(states, trajectory):
    X.append(np.hstack((state, ref)))
    Y.append([v, np.arctan2(ref[1] - state[1], ref[0] - state[0]) - state[2]])

X = np.array(X)
Y = np.array(Y)

# División de datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Salida [v, phi]
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))
