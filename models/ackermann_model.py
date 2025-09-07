import config
import numpy as np

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


class AckermannModel_Noise:
    def __init__(self, wheelbase=0.25, max_steer_angle=40):
        
        self.wheelbase = wheelbase
        
        self.max_steer_angle = np.radians(max_steer_angle)
        self.max_velocity = config.vmax

        # Ruidos
        self.velocity_noise_scale = config.velocity_noise_scale  # Ruido proporcional a la velocidad
        self.steering_noise_scale = config.steering_noise_scale  # Ruido proporcional al ángulo
        self.odometry_noise_scale = config.odometry_noise_scale  # Ruido en odometría

    def update(self, x, y, theta, velocity, steer_angle, dt):

        # Ruido en los COMANDOS (entradas del controlador) --------
        velocity_noise = np.random.normal(0, self.velocity_noise_scale * abs(velocity))
        steer_noise = np.random.normal(0, self.steering_noise_scale * abs(steer_angle))
        velocity_cmd = velocity + velocity_noise
        steer_cmd = steer_angle + steer_noise

        # Saturar comandos ruidosos
        steer_cmd = np.clip(steer_cmd, -self.max_steer_angle, self.max_steer_angle)
        velocity_cmd = np.clip(velocity_cmd, 0, self.max_velocity)


        # Ruido en la ODOMETRÍA (modelo interno) --------
        actual_velocity = velocity_cmd * (1 + np.random.normal(0, self.odometry_noise_scale))
        actual_steer = steer_cmd * (1 + np.random.normal(0, self.odometry_noise_scale))

        
        # Actualizar cinemática --------
        x_new = x + actual_velocity * np.cos(theta) * dt
        y_new = y + actual_velocity * np.sin(theta) * dt
        theta_new = theta + (actual_velocity / self.wheelbase) * np.tan(actual_steer) * dt

        return x_new, y_new, theta_new