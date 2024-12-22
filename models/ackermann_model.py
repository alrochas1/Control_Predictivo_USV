import numpy as np

def ackermann_model(state, inputs, dt):
    x, y, theta = state
    v, phi = inputs
    l = 2.5  # Distancia entre ejes (m)

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v * np.tan(phi) / l

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    return np.array([x, y, theta])