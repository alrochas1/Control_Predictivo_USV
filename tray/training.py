import numpy as np

def sin_tray(dt, T):
    time = np.linspace(0, T, int(T/dt))
    x_ref = time
    y_ref = np.sin(time / 4)
    theta_ref = np.arctan(np.gradient(y_ref, x_ref))  # Aproximación de orientación
    return np.vstack((x_ref, y_ref, theta_ref)).T


def circle_tray(dt, T, radio=1):
    time = np.linspace(0, T, int(T/dt))
    x_ref = radio * np.cos(time)
    y_ref = radio * np.sin(time)
    theta_ref = np.arctan2(np.gradient(y_ref, time), np.gradient(x_ref, time))
    return np.vstack((x_ref, y_ref, theta_ref)).T


def square_tray(dt, T, a=0.004, b=0, c=0):
    time = np.linspace(0, T, int(T/dt))
    x_ref = time
    y_ref = a * x_ref**2 + b * x_ref + c
    theta_ref = np.arctan2(np.gradient(y_ref, time), np.gradient(x_ref, time))  # Aproximación de orientación
    return np.vstack((x_ref, y_ref, theta_ref)).T