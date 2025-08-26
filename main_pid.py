# Este es para probar el controlador tradicional

import numpy as np

import config
from gym.env import AckermannTrakingEnv
from controller.pid_controller import PIDController


env = AckermannTrakingEnv()
controller = PIDController(config.kp, config.ki, config.kd)

obs, _ = env.reset()
    

for i in range(1000):
    lateral_error, orientation_error = obs

    steer_angle = controller.compute_control(lateral_error, orientation_error, env.dt)
    delta = np.clip(steer_angle, -config.max_steer, config.max_steer)

    v = 1.0

    action = np.array([v, np.radians(delta)], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n step={i}, error=({lateral_error:.3f},{orientation_error:.3f}), reward={reward:.3f}")

    x, y, theta = env.vehicle_pose
    print(f" Pos=({x:.3f},{y:.3f},{theta:.3f}), v={v:.3f}, delta(º)={delta:.3f}")
    env.render()

    if terminated or truncated:
        break

env.close()
