import numpy as np

def calculate_error(x, y, theta, x_target, y_target):

    error_position = np.sqrt((x_target - x)**2 + (y_target - y)**2)
    
    desired_theta = np.arctan2(y_target - y, x_target - x)
    
    error_orientation = desired_theta - theta
    error_orientation = (error_orientation + np.pi) % (2 * np.pi) - np.pi
    
    return error_position, error_orientation

# def calculate_error(predicted_states, desired_trajectory):

#     predicted_states = np.array(predicted_states)
#     desired_trajectory = np.array(desired_trajectory)

#     error = predicted_states - desired_trajectory

#     # Calcula el MSE (Error Cuadrático Medio)
#     mse = np.mean(error**2)
#     return mse
