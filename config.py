import numpy as np

# Parametros del modelo
L = 0.3
max_steer = 50
max_steer_rad = np.radians(max_steer)
vmax = 2.5
vmin = 0.0

# Parámetros para la generación de la trayectoria
dt = 0.01
N = 7   # N >= 3
x_range = (0, 40)  # Rango en el eje x
y_range = (-8, 8)  # Rango en el eje y


# Parametros para el control
kp = 5.0
ki = 10.0
kd = 0.1

max_lateral_error = 20

waypoint_threshold = 0.3 # metros para considerar que se alcanzó un waypoint


# Trayectoria Predefinida para comparativas

tray_x = [0,  8, 10, 32, 40, 25, 15, 8]
tray_y = [0,-12, 10, 12,  0,-10, -5, 8]