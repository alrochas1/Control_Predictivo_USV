# Este es para probar el controlador tradicional

import numpy as np

import config
from gym.env import AckermannTrakingEnv
from controller.pid_controller import PIDController
from tray.tracker import extract_metrics  # Función para extraer métricas y gráficas


env = AckermannTrakingEnv()
controller = PIDController(config.kp, config.ki, config.kd)

obs, _ = env.reset()

# Estadísticas
all_rewards = []  # Este aqui no tiene sentido, es para el otro
all_steps = []

i = 0
done = False
total_reward = 0
steps = 0

while not done:
    lateral_error, orientation_error = obs

    steer_angle = controller.compute_control(lateral_error, orientation_error, env.dt)
    delta = np.clip(steer_angle, -config.max_steer, config.max_steer)

    v = 1.0

    action = np.array([v, np.radians(delta)], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    i += 1

    print(f"\n step={i}, error=({lateral_error:.3f},{orientation_error:.3f}), reward={reward:.3f}")

    x, y, theta = env.vehicle_pose
    print(f" Pos=({x:.3f},{y:.3f},{theta:.3f}), v={v:.3f}, delta(º)={delta:.3f}")
    env.render()

    if terminated or truncated:
        break


all_rewards.append(total_reward)
all_steps.append(steps)

# Guarda Resultados
extract_metrics(env, 1, True)

env.close()
