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
        self.max_steps = 50/config.dt
        self.current_step = 0
        self.x_final, self.y_final, _ = self.reference_trajectory[-1]
        self.steering_turn_penalty = 0
        
        # Estado inicial (se establece correctamente en reset())
        self.state = None # [error_lateral, error_orientacion]
        self.vehicle_pose = None # [x, y, theta]
        
        # Para Pure Pursuit
        self.current_waypoint_index = 0
        self.lookahead_distance = 1.0  # metros para buscar el siguiente punto
        self.waypoint_threshold = config.waypoint_threshold

        # Históricos para métricas
        self.speed_history = []
        self.lateral_error_history = []
        self.orientation_error_history = []
        self.reward_history = []
        self.steering_history = []
        self.posx_history = []
        self.posy_history = []


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
        # print(f"Distance to target: {distance:.3f} m")
        
        # 2. ERROR LATERAL (Cross-track error): 
        # Proyección del vector en la dirección perpendicular a la orientación actual
        # Esto nos dice "qué tan a la izquierda o derecha de mi camino está el objetivo"
        tangent_vector = np.array([np.cos(theta_target), np.sin(theta_target)])
        normal_vector = np.array([-np.sin(theta_target), np.cos(theta_target)])
        
        # Vector desde punto objetivo al vehículo
        vehicle_to_target = np.array([dx, dy])
        
        # Proyección sobre la normal (ésta es la distancia lateral REAL)
        lateral_error = np.dot(vehicle_to_target, normal_vector)
        

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
            self.current_waypoint_index = min(self.current_waypoint_index + 1, 
                                    len(self.reference_trajectory) - 1)
        
        
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
        self.steering_turn_penalty = 0
        self.prev_steer = 0.0
        self.prev_progress = 0.0

        # Resetear históricos
        self.speed_history = []
        self.lateral_error_history = []
        self.orientation_error_history = []
        self.reward_history = []
        self.steering_history = []
        self.posx_history = []
        self.posy_history = []

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
        x, y, theta = self.vehicle.update(x_old, y_old, theta_old, v, delta, self.dt)
        self.vehicle_pose = np.array([x, y, theta])
        # print(f"Posicion: ({new_x:.2f}, {new_y:.2f}, {new_theta:.2f}), Velocidad: {v:.2f}")

        # 3. Calcular la nueva observación (los nuevos errores)
        lateral_error, orientation_error = self.calculate_errors()
        self.state = np.array([lateral_error, orientation_error], dtype=np.float32)

        # DEBUG
        # if abs(orientation_error) > 1.0:  # Si hay un salto grande
            # print(f"¡SALTO DETECTADO!")
            # print(f"Estado actual: {self.state}")
            # print(f"Waypoint anterior: {old_waypoint_index}, Target: {old_target}")
            # print(f"Waypoint nuevo: {self.current_waypoint_index}")
            # print(f"Error normalizado: {orientation_error}")
        # END DEBUG

        # ===================================================
        # ============= SISTEMAS DE RECOMPENSAS =============
        # ===================================================

        # 1. RECOMPENSA BASE por velocidad (incentivar movimiento)
        base_speed_reward = 1.0 * v * self.dt  # Recompensa proporcional al tiempo
        if v < 0.1:
            base_speed_reward -= 50.0 * self.dt  # Penalización por estar parado

        # 2. RECOMPENSA por SEGUIMIENTO PRECISO (exponencial inversa al error)
        tracking_reward = (
            np.exp(-0.5  * abs(lateral_error)) + 
            np.exp(-2.0 * abs(orientation_error))
        ) * 10*self.dt

        # 3. PENALIZACIÓN por CAMBIOS BRUSCOS de dirección (suavidad del control)
        jerk_penalty = 0.0
        if hasattr(self, 'prev_steer'):
            steer_change = abs(action[1] - self.prev_steer)
            jerk_penalty = 10.0 * steer_change * self.dt  # Escalado por dt
        self.prev_steer = action[1]
        
        # 4. RECOMPENSA por PROGRESO en la trayectoria
        current_progress = self.current_waypoint_index / len(self.reference_trajectory)
        progress_reward = 0.0
        if hasattr(self, 'prev_progress'):
            progress_delta = current_progress - self.prev_progress
            progress_reward = 10.0 * progress_delta  # Grande por avanzar
        self.prev_progress = current_progress

        # 5. PENALIZACIÓN por GIRO CONTINUO EXCESIVO
        extended_turn_penalty = 0.0
        # if hasattr(self, 'steering_history'):
        #     # 5 segundos de historial
        #     samples_to_check = int(5.0 / self.dt)
        #     recent_steering = self.steering_history[-samples_to_check:] 
            
        #     # Mínimo 1 segundo de datos para evaluar
        #     if len(recent_steering) >= int(1.0 / self.dt):
        #         large_steering = all(abs(s) > 0.7 for s in recent_steering)
                
        #         if large_steering:
        #             extended_turn_penalty = 0*100.0  # Penalización continua
        #             if self.current_step % int(1 / self.dt) == 0:  # Log cada 1 segundo
        #                 print("⚠️  Penalización: Giro continuo prolongado")

        # 6. RECOMPENSA por VELOCIDAD ADAPTADA a la CURVATURA
        curvature_reward = 0.0
        if self.current_waypoint_index < len(self.reference_trajectory) - 1:
            current_point = self.reference_trajectory[self.current_waypoint_index]
            next_point = self.reference_trajectory[self.current_waypoint_index + 1]
            
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            trajectory_curvature = abs(dy) / (abs(dx) + 1e-6)
            
            # Velocidad deseada en función de la curvatura
            desired_speed = 2.0 * np.exp(-2.0 * trajectory_curvature)
            
            # Recompensa por ajustarse a la velocidad deseada
            speed_error = abs(v - desired_speed)
            curvature_reward = 5 * np.exp(-2.0 * speed_error) * self.dt
            
            # Debug info cada 5 segundos de simulación
            # if self.current_step % int(5.0 / self.dt) == 0:
            #     print(f"🔄 Curvatura: {trajectory_curvature:.2f}, "
            #         f"Velocidad deseada: {desired_speed:.2f}m/s, "
            #         f"Actual: {v:.2f}m/s, Reward: {curvature_reward:.2f}")
                
        # 7. CONDICIONES DE TERMINACIÓN con RECOMPENSAS FINALES
        terminated = False
        truncated = False
        completion_bonus = 0.0
        failure_penalty = 0.0

        # Terminar por desviación excesiva
        if abs(lateral_error) > 4.0:
            failure_penalty = 50.0  # Penalización fija
            terminated = True
            print("¡Desviación excesiva detectada!")

        # Terminar por completar la trayectoria
        distance_to_end = np.sqrt((self.x_final - x)**2 + (self.y_final - y)**2)
        if distance_to_end < 0.1: 
            completion_bonus = 100.0  # Recompensa fija final
            terminated = True
            print(f"¡Trayectoria completada con éxito! "
                f"Distancia al final: {distance_to_end:.2f}m")
            

        # ===== RECOMPENSA TOTAL =====
        reward = (
            base_speed_reward +
            tracking_reward +
            progress_reward +
            curvature_reward +
            completion_bonus -
            jerk_penalty -
            extended_turn_penalty -
            failure_penalty
        )
        # print(f"Step {self.current_step}: Reward: {reward:.2f} = "
        #       f"{base_speed_reward:.2f} + {tracking_reward:.2f} + {progress_reward:.2f} + "
        #       f"{curvature_reward:.2f} + {completion_bonus:.2f} - {jerk_penalty:.2f} - "
        #       f"{extended_turn_penalty:.2f} - {failure_penalty:.2f}")

        # ===================================================
        # ===================================================
            
        # 5. Comprobar condiciones de terminación (truncated es por límite de pasos)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 6. Información de debug (opcional)
        info = {
            "pose": self.vehicle_pose,
            "action": action,
            "tracking_reward": tracking_reward,
            "current_waypoint": self.current_waypoint_index,
            "total_waypoints": len(self.reference_trajectory),
            "progress": current_progress,
            "lateral_error": lateral_error,
            "orientation_error": orientation_error
        }
        
        # Guardar métricas
        self.speed_history.append(v)
        self.lateral_error_history.append(lateral_error)
        self.orientation_error_history.append(orientation_error)
        self.reward_history.append(reward)
        self.steering_history.append(np.degrees(action[1]))  # En grados
        self.posx_history.append(x)
        self.posy_history.append(y)

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


    def get_metrics(self):
        """Devuelve métricas resumidas del episodio"""
        if not self.speed_history:
            return {}
            
        return {
            'mean_speed': np.mean(self.speed_history),
            'max_speed': np.max(self.speed_history),
            'min_speed': np.min(self.speed_history),
            'mean_lateral_error': np.mean(np.abs(self.lateral_error_history)),
            'max_lateral_error': np.max(np.abs(self.lateral_error_history)),
            'mean_orientation_error': np.mean(np.abs(self.orientation_error_history)),
            'total_reward': np.sum(self.reward_history),
            'mean_steering': np.mean(np.abs(self.steering_history)),
            'episode_length': len(self.speed_history)
        }
    

    # Genera gráficas de las métricas
    def plot_metrics(self, save_path=None):
        if not self.speed_history:
            print("No hay datos para graficar")
            return
            
        steps = range(len(self.speed_history))
        title_fontsize = 18
        label_fontsize = 15
        legend_fontsize = 13
        tick_fontsize = 13

        # Figura 1: Errores y posiciones en una cuadrícula 2x2
        fig1, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Gráfica 1: Error Lateral
        axs[0, 0].plot(steps, self.lateral_error_history, 'g-', label='Error Lateral (m)')
        axs[0, 0].set_xlabel('Steps', fontsize=label_fontsize)
        axs[0, 0].set_ylabel('Error Lateral (m)', fontsize=label_fontsize)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Error Lateral', fontsize=title_fontsize)
        axs[0, 0].legend(fontsize=legend_fontsize)
        axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 2: Error Orientación
        axs[0, 1].plot(steps, np.degrees(self.orientation_error_history), 'm-', label='Error Orientación (°)')
        axs[0, 1].set_xlabel('Steps', fontsize=label_fontsize)
        axs[0, 1].set_ylabel('Error Orientación (°)', fontsize=label_fontsize)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Error de Orientación', fontsize=title_fontsize)
        axs[0, 1].legend(fontsize=legend_fontsize)
        axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 3: Posición en X
        axs[1, 0].plot(steps, self.posx_history, 'b-', label='Posición X (m)')
        axs[1, 0].set_xlabel('Steps', fontsize=label_fontsize)
        axs[1, 0].set_ylabel('Posición X (m)', fontsize=label_fontsize)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Posición en X', fontsize=title_fontsize)
        axs[1, 0].legend(fontsize=legend_fontsize)
        axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 4: Posición en Y
        axs[1, 1].plot(steps, self.posy_history, 'c-', label='Posición Y (m)')
        axs[1, 1].set_xlabel('Steps', fontsize=label_fontsize)
        axs[1, 1].set_ylabel('Posición Y (m)', fontsize=label_fontsize)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Posición en Y', fontsize=title_fontsize)
        axs[1, 1].legend(fontsize=legend_fontsize)
        axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)

        plt.tight_layout()
        if save_path:
            fig1.savefig(save_path.replace('.png', '_errores.png'))
            print(f"Gráficas de errores guardadas en {save_path.replace('.png', '_errores.png')}")

        # Figura 2: Métricas varias
        fig2, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Gráfica 1: Recompensa acumulada
        cumulative_reward = np.cumsum(self.reward_history)
        axs[0, 0].plot(steps, cumulative_reward, 'purple')
        axs[0, 0].set_xlabel('Steps', fontsize=label_fontsize)
        axs[0, 0].set_ylabel('Recompensa Acumulada', fontsize=label_fontsize)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Recompensa Acumulada', fontsize=title_fontsize)
        axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 2: Histograma de velocidades
        axs[0, 1].hist(self.speed_history, bins=20, alpha=0.7, color='orange')
        axs[0, 1].set_xlabel('Velocidad (m/s)', fontsize=label_fontsize)
        axs[0, 1].set_ylabel('Frecuencia', fontsize=label_fontsize)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Distribución de Velocidades', fontsize=title_fontsize)
        axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 3: Velocidad
        axs[1, 0].plot(steps, self.speed_history, 'b-', label='Velocidad (m/s)')
        axs[1, 0].set_xlabel('Steps', fontsize=label_fontsize)
        axs[1, 0].set_ylabel('Velocidad (m/s)', fontsize=label_fontsize)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Velocidad', fontsize=title_fontsize)
        axs[1, 0].legend(fontsize=legend_fontsize)
        axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)

        # Gráfica 4: Steering
        axs[1, 1].plot(steps, self.steering_history, 'r-', label='Steering (°)')
        axs[1, 1].set_xlabel('Steps', fontsize=label_fontsize)
        axs[1, 1].set_ylabel('Ángulo de dirección (°)', fontsize=label_fontsize)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Steering', fontsize=title_fontsize)
        axs[1, 1].legend(fontsize=legend_fontsize)
        axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)

        plt.tight_layout()
        if save_path:
            fig2.savefig(save_path.replace('.png', '_metricas.png'))
            print(f"Gráficas de métricas guardadas en {save_path.replace('.png', '_metricas.png')}")

        plt.show()