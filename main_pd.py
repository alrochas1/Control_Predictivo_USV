import numpy as np
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import config
from tray import aux

from models.ackermann_model import AckermannModel as model
from controller.pd_controller import PDController
from error import calculate_error


# Parámetros para la generación de la trayectoria
dt = config.dt
N = config.N   # N >= 3
x_range = config.x_range  # Rango en el eje x
y_range = config.y_range  # Rango en el eje y


# Generar la trayectoria
tray = aux.random_bezier_trajectory(N, x_range, y_range)
x_traj = tray[:, 0]
y_traj = tray[:, 1]



# Parámetros del controlador PD
kp = 40.0
kd = 0.1

pd_controller = PDController(kp, kd)
vehicle = model(config.L, config.max_steer)

# Condiciones iniciales del vehículo
x, y = 0.0, 0.0
theta = aux.compute_start(tray)*0   # REVISAR
velocity = 0.05


x_vehicle, y_vehicle = [0], [0]
t, t_error_pos, t_error_ori = [], [], []

threshold = 0.3  # Umbral de proximidad
current_point, i = 0, 0

while current_point < len(x_traj):

    x_target = x_traj[current_point]
    y_target = y_traj[current_point]
    
    error_position, error_orientation = calculate_error(x, y, theta, x_target, y_target)
    steer_angle = pd_controller.compute_control(error_position, error_orientation, dt)
    
    x, y, theta = vehicle.update(x, y, theta, velocity, steer_angle, dt)
    x_vehicle.append(x)
    y_vehicle.append(y)
    t_error_pos.append(error_position)
    t_error_ori.append(error_orientation)
    t.append(i*dt)
    i += 1
    
    # Calcular la distancia al punto actual
    distance_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2)
    if distance_to_target < threshold:
        current_point += 1

    # Para evitar que se quede pillado
    if i > 50*len(x_traj):
        current_point = len(x_traj)



plt.figure()
plt.plot(x_traj, y_traj, label="Trayectoria deseada")
plt.plot(x_vehicle, y_vehicle, 'r', label="Trayectoria seguida")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.title("Seguimiento de Trayectoria con Control PD")


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, t_error_pos, label="Evolución del error en la posición", color='b')
plt.xlabel("t(s)")
plt.ylabel("MSE")
plt.legend()
plt.title("Evolución del error en la posición con el tiempo")

plt.subplot(2, 1, 2)
plt.plot(t, t_error_ori, label="Evolución del error en la orientación", color='r')
plt.xlabel("t(s)")
plt.ylabel("MSE")
plt.legend()
plt.title("Evolución del error en la orientación con el tiempo")

plt.tight_layout()
plt.show()


