import numpy as np

# Controlador PD para el ángulo de dirección
def pd_controller(current_state, target_state, Kp=2.0, Kd=0.1):
    error_lateral = np.sqrt((target_state[0] - current_state[0])**2 + 
                            (target_state[1] - current_state[1])**2)
    error_heading = np.arctan2(target_state[1] - current_state[1], 
                               target_state[0] - current_state[0]) - current_state[2]
    
    # Derivada (solo el error angular)
    phi = Kp * error_heading + Kd * error_lateral
    return np.clip(phi, -np.pi/4, np.pi/4)