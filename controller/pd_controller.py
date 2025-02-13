# Controlador PD para el ángulo de dirección
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp 
        self.kd = kd 
        # self.prev_error = 0 
        self.prev_error_position = 0
        self.prev_error_orientation = 0

        self.orientation_threshold = 0.02


    def compute_control(self, error_position, error_orientation, dt):
        
        if abs(error_orientation) > self.orientation_threshold:
            # Priorizar la alineación (control de orientación)
            p_term = self.kp * error_orientation
            d_term = self.kd * (error_orientation - self.prev_error_orientation) / dt
            self.prev_error_orientation = error_orientation
        else:
            # Priorizar el avance hacia el punto objetivo (control de posición)
            p_term = self.kp * error_position
            d_term = self.kd * (error_position - self.prev_error_position) / dt
            self.prev_error_position = error_position
        
        steer_angle = p_term + d_term
        
        return steer_angle