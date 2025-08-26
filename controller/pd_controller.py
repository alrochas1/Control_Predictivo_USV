# Controlador PD para el ángulo de dirección
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp 
        self.kd = kd 
        self.prev_error_lateral = 0
        self.prev_error_orientation = 0

        self.orientation_threshold = 0.1

    
    def compute_control(self, lateral_error, orientation_error, dt):
        if abs(orientation_error) > self.orientation_threshold:
            # Priorizar la ALINEACIÓN (estamos muy mal orientados)
            p_term = self.kp * orientation_error
            d_term = self.kd * (orientation_error - self.prev_error_orientation) / dt
            self.prev_error_orientation = orientation_error
            self.prev_error_lateral = lateral_error  # Resetear para evitar derivada spikes
        else:
            # Priorizar el SEGUIMIENTO LATERAL (ya estamos bien orientados)
            p_term = self.kp * lateral_error
            d_term = self.kd * (lateral_error - self.prev_error_lateral) / dt
            self.prev_error_lateral = lateral_error
            self.prev_error_orientation = orientation_error  # Resetear
        
        steer_angle = p_term + d_term
        return steer_angle