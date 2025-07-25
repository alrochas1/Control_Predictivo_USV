import numpy as np
import tensorflow as tf

phi_max = 60*3.1415/180


class AckermannModel:
    def __init__(self, wheelbase=0.25, max_steer_angle=40):

        self.wheelbase = wheelbase
        self.max_steer_angle = np.radians(max_steer_angle)

    def update(self, x, y, theta, velocity, steer_angle, dt):
        
        steer_angle = np.clip(steer_angle, -self.max_steer_angle, self.max_steer_angle)
        velocity = np.clip(velocity, 0, 20)
        
        x_new = x + velocity * np.cos(theta) * dt
        y_new = y + velocity * np.sin(theta) * dt
        theta_new = theta + (velocity / self.wheelbase) * np.tan(steer_angle) * dt
        
        return x_new, y_new, theta_new


# Mismos modelos, pero definidos usando tensores
class AckermannModel_tf:
    def __init__(self, wheelbase=0.25, max_steer_angle=40):
        self.wheelbase = tf.constant(wheelbase, dtype=tf.float32)
        self.max_steer_angle = tf.constant(float(max_steer_angle) * (np.pi / 180), dtype=tf.float32)

    def update(self, x, y, theta, velocity, steer_angle, dt):
        steer_angle = tf.clip_by_value(steer_angle, -self.max_steer_angle, self.max_steer_angle)
        velocity = tf.clip_by_value(velocity, 0, 20)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        theta = tf.cast(theta, tf.float32)

        x_new = x + velocity * tf.cos(theta) * dt
        y_new = y + velocity * tf.sin(theta) * dt
        theta_new = theta + (velocity / self.wheelbase) * tf.tan(steer_angle) * dt

        return tf.stack([x_new, y_new, theta_new])  # Devuelve un tensor



def ackermann_model(state, inputs, dt):
    x, y, theta = state
    v, phi = inputs
    l = 2.5  # Distancia entre ejes (m)

    # Saturación (giro maximo)
    phi = np.clip(phi, -phi_max, phi_max)

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
