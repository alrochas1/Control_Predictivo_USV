import numpy as np

phi_max = 60*3.1415/180

def ackermann_model(state, inputs, dt):
    x, y, theta = state
    v, phi = inputs
    l = 2.5  # Distancia entre ejes (m)

    # Saturación (giro maximo)
    # phi = np.clip(phi, -phi_max, phi_max)

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v * np.tan(phi) / l

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    return np.array([x, y, theta])


def ackermann_model_noise(state, inputs, dt, noise_std_inputs=0.01, noise_std_states=0.005):

    # noise_std_inputs = Simula errores en los actuadores
    # noise_std_states = Simula errores en sensores

    x, y, theta = state
    v, phi = inputs
    l = 2.5  # Distancia entre ejes (m)

    # Agregar ruido a las entradas
    v_noisy = v + np.random.normal(0, noise_std_inputs)
    phi_noisy = phi + np.random.normal(0, noise_std_inputs)

    # Saturación (giro maximo)
    phi_noisy = np.clip(phi_noisy, -phi_max, phi_max)

    # Dinámica del modelo
    x_dot = v_noisy * np.cos(theta)
    y_dot = v_noisy * np.sin(theta)
    theta_dot = v_noisy * np.tan(phi_noisy) / l

    # Integración de los estados
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    # Agregar ruido al estado
    x_noisy = x + np.random.normal(0, noise_std_states)
    y_noisy = y + np.random.normal(0, noise_std_states)
    theta_noisy = theta + np.random.normal(0, noise_std_states)

    return np.array([x_noisy, y_noisy, theta_noisy])
