import numpy as np
import matplotlib.pyplot as plt

from models.ackermann_model import ackermann_model


# Controlador PD para el ángulo de dirección
def pd_controller(current_state, target_state, Kp=2.0, Kd=0.1):
    error_lateral = np.sqrt((target_state[0] - current_state[0])**2 + 
                            (target_state[1] - current_state[1])**2)
    error_heading = np.arctan2(target_state[1] - current_state[1], 
                               target_state[0] - current_state[0]) - current_state[2]
    
    # Derivada (solo el error angular)
    phi = Kp * error_heading + Kd * error_lateral
    return np.clip(phi, -np.pi/4, np.pi/4)  # Límite del ángulo de dirección

# Parámetros de simulación
dt = 0.1  # Paso de tiempo (s)
T = 20    # Tiempo total (s)
steps = int(T / dt)

# Trayectoria deseada (curva sinusoidal)
time = np.linspace(0, T, steps)
x_ref = time
y_ref = np.sin(time / 5)
theta_ref = np.arctan(np.gradient(y_ref, x_ref))  # Aproximación de orientación
trajectory = np.vstack((x_ref, y_ref, theta_ref)).T

# Simulación del modelo Ackermann
state = np.array([0, 0, 0])  # Estado inicial [x, y, theta]
v = 1.0  # Velocidad constante (m/s)
states = [state]

for i in range(steps - 1):
    target_state = trajectory[i + 1]
    phi = pd_controller(state, target_state)
    state = ackermann_model(state, [v, phi], dt)
    states.append(state)

states = np.array(states)

# Visualización
plt.plot(x_ref, y_ref, label='Trayectoria Deseada', linestyle='--')
plt.plot(states[:, 0], states[:, 1], label='Trayectoria Seguida (PD Control)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación con Controlador PD')
plt.show()
