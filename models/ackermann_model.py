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
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # Normalizar entre [-pi, pi]
        
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
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # Normalizar entre [-pi, pi]

        return tf.stack([x_new, y_new, theta_new])  # Devuelve un tensor