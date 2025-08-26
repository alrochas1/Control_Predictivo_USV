import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

import config
from tray import aux
from models.ackermann_model import AckermannModel as model


class AckermannTrakingEnv(gym.Env):

    def __init__(self):
        super(AckermannTrakingEnv, self).__init__()

        # Modelo del vehiculo
        self.vehicle = model(config.L, config.max_steer)
        
        # Definir el espacio de ACCIONES que puede tomar el agente.
        # [velocidad, ángulo de dirección]
        self.action_space = spaces.Box(
            low = np.array([config.vmin, -config.max_steer_rad], dtype=np.float32),
            high= np.array([config.vmax, config.max_steer_rad],  dtype=np.float32),
            dtype=np.float32
        )
        
        # Definir el espacio de OBSERVACIONES (estado) que recibe el agente.
        # [error_lateral, error_orientacion]
        self.observation_space = spaces.Box(
            low=np.array([-config.max_lateral_error, -3.14], dtype=np.float32),
            high=np.array([config.max_lateral_error, 3.14],  dtype=np.float32),
            dtype=np.float32
        )
        
        # Inicializar el estado de la simulación
        self.reference_trajectory = aux.get_reference_trajectory()
        self.dt = config.dt
        self.max_steps = 500/config.dt
        self.current_step = 0
        
        # Estado inicial (se establece correctamente en reset())
        self.state = None # [error_lateral, error_orientacion]
        self.vehicle_pose = None # [x, y, theta]
        
        # Para Pure Pursuit
        self.current_waypoint_index = 0
        self.lookahead_distance = 1.0  # metros para buscar el siguiente punto
        self.waypoint_threshold = config.waypoint_threshold


    # Calcula errores usando lógica Pure Pursuit: busca un punto adelante
    # en la trayectoria y calcula los errores respecto a ese punto.
    def calculate_errors(self):
        if self.current_waypoint_index >= len(self.reference_trajectory):
            # Si llegamos al final, usar el último punto
            target_point = self.reference_trajectory[-1]
        else:
            target_point = self.reference_trajectory[self.current_waypoint_index]
        
        x_target, y_target, theta_target = target_point
        x, y, theta = self.vehicle_pose
    
        # 1. Vector desde el coche al objetivo
        dx = x_target - x
        dy = y_target - y
        distance = np.sqrt((x_target - x)**2 + (y_target - y)**2)
        print(f"Distance to target: {distance:.3f} m")
        
        # 2. ERROR LATERAL (Cross-track error): 
        # Proyección del vector en la dirección perpendicular a la orientación actual
        # Esto nos dice "qué tan a la izquierda o derecha de mi camino está el objetivo"
        lateral_error = -dx * np.sin(theta) + dy * np.cos(theta)
        
        # 3. ERROR DE ORIENTACIÓN: 
        # Ángulo hacia el objetivo menos orientación actual
        if distance > 0.1:  # Si estamos suficientemente lejos
            # Calcular geométricamente (normal)
            desired_theta = np.arctan2(y_target - y, x_target - x)
            orientation_error = aux.angle_difference(desired_theta, theta)
        else:
            # Si estamos MUY CERCA, usar la orientación del waypoint directamente
            orientation_error = aux.angle_difference(theta_target, theta)

        # Normalizar a [-pi, pi]
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi
        
        return lateral_error, orientation_error
    

    # Actualiza el índice del waypoint actual si se esta lo suficientemente cerca.
    def update_waypoint_index(self):
        if self.current_waypoint_index >= len(self.reference_trajectory) - 1:
            return  # Ya estamos en el último waypoint
        
        x, y, _ = self.vehicle_pose
        current_target = self.reference_trajectory[self.current_waypoint_index]
        x_target, y_target, _ = current_target
        
        distance_to_target = np.sqrt((x_target - x)**2 + (y_target - y)**2)
        
        if distance_to_target < self.waypoint_threshold:
            self.current_waypoint_index += 1
        
        
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # Reinicia el contador de pasos
        self.current_step = 0
        self.current_waypoint_index = 0
        self.vehicle_path = []
        
        # Coloca el vehículo en su estado inicial
        start_x, start_y, start_theta = self.reference_trajectory[0]
        self.vehicle_pose = np.array([
            start_x + 0.5*0,
            start_y - 0.5*0,
            start_theta
        ])
        
        # Calcula la observación inicial (errores)
        lateral_error, orientation_error = self.calculate_errors()
        self.state = np.array([lateral_error, orientation_error], dtype=np.float32)

        return self.state, {}

    
    def step(self, action):
        # 1. Parsear la acción: [velocidad (v), ángulo de dirección (delta)]
        v, delta = action

        # DEBUG
        old_waypoint_index = self.current_waypoint_index
        old_target = self.reference_trajectory[old_waypoint_index]
        # END DEBUG

        # 2. Actualizar el waypoint si es necesario
        self.update_waypoint_index()
        
        # 2. Aplicar la acción al modelo cinemático del vehículo
        # (Usa TU función aquí)
        x_old, y_old, theta_old = self.vehicle_pose
        new_x, new_y, new_theta = self.vehicle.update(x_old, y_old, theta_old, v, delta, self.dt)
        self.vehicle_pose = np.array([new_x, new_y, new_theta])
        # print(f"Posicion: ({new_x:.2f}, {new_y:.2f}, {new_theta:.2f}), Velocidad: {v:.2f}")

        # 3. Calcular la nueva observación (los nuevos errores)
        lateral_error, orientation_error = self.calculate_errors()
        self.state = np.array([lateral_error, orientation_error], dtype=np.float32)

        # DEBUG
        if abs(orientation_error) > 1.0:  # Si hay un salto grande
            print(f"¡SALTO DETECTADO!")
            print(f"Estado actual: {self.state}")
            print(f"Waypoint anterior: {old_waypoint_index}, Target: {old_target}")
            print(f"Waypoint nuevo: {self.current_waypoint_index}")
            print(f"Error normalizado: {orientation_error}")
        # END DEBUG
        
        # 4. Calcular la RECOMPENSA (PENDIENTE DE HACER)
        # - Recompensa base por avanzar (incentiva la velocidad positiva)
        reward = v * self.dt
        # - Penalización por errores (¡Fórmula clave!)
        reward -= 0.5 * abs(lateral_error)  # Penaliza el error lateral
        reward -= 0.1 * abs(orientation_error) # Penaliza el error de orientación
        # - Penalización MUY FUERTE por salirse de la carretera
        # if abs(lateral_error) > 2.0: # Si se desvía más de 2 metros
        #     reward -= 10.0
        #     terminated = True
        # else:
        #     terminated = False
        terminated = False

        # Terminar si llegamos al final de la trayectoria
        if self.current_waypoint_index >= len(self.reference_trajectory) - 1:
            # Verificar si estamos cerca del punto final
            final_point = self.reference_trajectory[-1]
            x, y, _ = self.vehicle_pose
            distance_to_end = np.sqrt((final_point[0] - x)**2 + (final_point[1] - y)**2)
            if distance_to_end < self.waypoint_threshold:
                reward += 100.0  # Gran recompensa por completar la trayectoria
                terminated = True
            
        # 5. Comprobar condiciones de terminación (truncated es por límite de pasos)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 6. Información de debug (opcional)
        info = {
            "pose": self.vehicle_pose,
            "action": action,
            "current_waypoint": self.current_waypoint_index,
            "total_waypoints": len(self.reference_trajectory)
        }
        
        # 7. Devolver la tupla (obs, reward, terminated, truncated, info)
        return self.state, reward, terminated, truncated, info


    def render(self, mode="human"):
        # Inicializar la figura la primera vez
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            self.ax.plot(self.reference_trajectory[:,0],
                         self.reference_trajectory[:,1],
                         'k--', label="trayectoria ref")
            self.ax.axis("equal")
            self.ax.legend()
            plt.ion()
            plt.show()
            
            # Flecha (pose del coche)
            self.arrow = None
            # Historial de poses
            self.trajectory_x = []
            self.trajectory_y = []

        # Guardar la pose actual
        x, y, theta = self.vehicle_pose
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)

        # Borrar flecha previa
        if self.arrow is not None:
            self.arrow.remove()
        # Dibujar flecha nueva (coche)
        arrow_length = 0.5
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        self.arrow = FancyArrow(x, y, dx, dy,
                                width=0.1, color="r")
        self.ax.add_patch(self.arrow)

        # Refrescar
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    # Llamar al final del episodio para mostrar la trayectoria seguida
    def close(self):
        if hasattr(self, 'trajectory_x'):
            plt.ioff()
            plt.figure()
            plt.plot(self.reference_trajectory[:,0],
                     self.reference_trajectory[:,1], 'k--', label="trayectoria ref")
            plt.plot(self.trajectory_x,
                     self.trajectory_y, 'r-', label="trayectoria seguida")
            plt.axis("equal")
            plt.legend()
            plt.title("Trayectoria seguida vs referencia")
            plt.show()