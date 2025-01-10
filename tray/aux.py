import numpy as np

def get_start_state(tray):
    x_start, y_start = tray[0, 0], tray[0, 1]
    x_next, y_next = tray[1, 0], tray[1, 1]
    initial_theta = np.arctan2(y_next - y_start, x_next - x_start)
    return initial_theta
