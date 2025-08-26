import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from gym.env import AckermannTrakingEnv  # Importa tu entorno


model_path="ackermann_ppo_model.zip"
num_episodes=1
render=True

print("Cargando modelo entrenado...")
try:
    model = PPO.load(model_path)
    print(f"Modelo cargado desde {model_path}")
except Exception as e:
    print(f"Error cargando modelo: {e}")


print("Creando entorno de prueba...")
env = AckermannTrakingEnv()

# Estadísticas
all_rewards = []
all_steps = []
successes = 0

for episode in range(num_episodes):
    print(f"\n=== EPISODIO {episode + 1}/{num_episodes} ===")
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Predecir acción con el modelo entrenado
        action, _states = model.predict(obs, deterministic=True)
        
        # Ejecutar acción en el entorno
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # Visualización
        if render and steps % 5 == 0:  # Render cada 5 steps
            env.render()
            
        # Print cada 50 steps
        if steps % 50 == 0:
            x, y, theta = env.vehicle_pose
            print(f"Step {steps}: Posición: ({x:.2f}, {y:.2f}, {theta:.2f}), \n \t Reward acumulado: {total_reward:.2f}, \n Velocidad: {action[0]:.2f}")

    # Resultados del episodio
    episode_result = "ÉXITO" if not truncated else "TIMEOUT" if steps >= env.max_steps else "FALLO"
    print(f"{episode_result} - Steps: {steps}, Reward total: {total_reward:.2f}")

    if episode_result == "ÉXITO":  # Completó sin fallos
        successes += 1
    
    all_rewards.append(total_reward)
    all_steps.append(steps)

# Estadísticas finales
print(f"\nESTADÍSTICAS FINALES:")
print(f"Episodios completados: {successes}/{num_episodes}")
print(f"Reward promedio: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
print(f"Steps promedio: {np.mean(all_steps):.2f} ± {np.std(all_steps):.2f}")

env.close()