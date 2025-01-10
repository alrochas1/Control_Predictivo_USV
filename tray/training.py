import numpy as np

def sin_tray(dt, T):
    time = np.linspace(0, T, int(T/dt))
    x_ref = time
    y_ref = np.sin(time / 5)
    theta_ref = np.arctan(np.gradient(y_ref, x_ref))  # Aproximación de orientación
    return np.vstack((x_ref, y_ref, theta_ref)).T