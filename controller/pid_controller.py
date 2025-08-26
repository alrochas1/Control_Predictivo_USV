# Controlador PID para el ángulo de dirección

import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error_lateral = 0
        self.prev_error_orientation = 0
        self.integral_lateral = 0.0
        self.integral_orientation = 0.0

        self.orientation_threshold = 0.1
        self.integral_limit = 1.0

    
    def compute_control(self, lateral_error, orientation_error, dt):
        if abs(orientation_error) > self.orientation_threshold:
            # Modo orientación
            p_term = self.kp * orientation_error
            d_term = self.kd * (orientation_error - self.prev_error_orientation) / dt
            
            # Accion Integral
            self.integral_orientation += orientation_error * dt
            self.integral_orientation = np.clip(self.integral_orientation, -self.integral_limit, self.integral_limit)
            i_term = self.ki * self.integral_orientation
            
            steer_angle = p_term + i_term + d_term
            
            # Resetear integral del modo contrario
            self.integral_lateral = 0.0
            self.prev_error_orientation = orientation_error
            
        else:
            # Modo seguimiento lateral  
            p_term = self.kp * lateral_error
            d_term = self.kd * (lateral_error - self.prev_error_lateral) / dt
            
            # Accion Integral
            self.integral_lateral += lateral_error * dt
            self.integral_lateral = np.clip(self.integral_lateral, -self.integral_limit, self.integral_limit)
            i_term = self.ki * self.integral_lateral
            
            steer_angle = p_term + i_term + d_term
            
            # Resetear integral del modo contrario
            self.integral_orientation = 0.0
            self.prev_error_lateral = lateral_error
        
        return steer_angle