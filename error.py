import numpy as np

def calculate_error(predicted_states, desired_trajectory):
    """
    Calcula el error promedio entre los estados predichos y la trayectoria deseada.
    
    Parameters:
    - predicted_states: np.ndarray, estados predichos por la red neuronal, forma (N, 3) -> [x, y, theta]
    - desired_trajectory: np.ndarray, trayectoria deseada, forma (N, 3) -> [x_ref, y_ref, theta_ref]
    
    Returns:
    - mse: float, error cuadrático medio (MSE)
    """
    # Error entre los puntos predichos y los deseados
    error = predicted_states - desired_trajectory
    # Calculamos el MSE (Error Cuadrático Medio)
    mse = np.mean(np.sum(error**2, axis=1))  # Error acumulado por cada estado [x, y, theta]
    return mse
