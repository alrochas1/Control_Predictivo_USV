import matplotlib.pyplot as plt
import numpy as np

import config
from gym.env import AckermannTrakingEnv
from stable_baselines3 import PPO
from matplotlib.gridspec import GridSpec
from controller.pid_controller import PIDController  # Importa tu PID

# Evalúa un controlador (PID o RL) en el entorno y devuelve las métricas
def evaluate_controller(env, controller_type, controller_obj, num_episodes=1, render=False):
    all_rewards = []
    all_steps = []
    all_poses = []
    all_actions = []
    all_errors = []
    successes = 0

    for episode in range(num_episodes):
        print(f"\n=== {controller_type} - EPISODIO {episode + 1}/{num_episodes} ===")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        episode_poses = []
        episode_actions = []
        episode_errors = []

        while not done:
            # Predecir acción según el tipo de controlador
            if controller_type == "RL":
                action, _states = controller_obj.predict(obs, deterministic=True)
            else:  # PID
                lateral_error, orientation_error = obs
                action = controller_obj.compute_control(lateral_error, orientation_error, env.dt)
                action = np.array([1.5, action])  # [velocidad, ángulo]
            
            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Guardar datos para análisis
            episode_poses.append(env.vehicle_pose.copy())
            episode_actions.append(action.copy())
            episode_errors.append(obs.copy())  # [lateral_error, orientation_error]
            
            # Visualización
            if render and steps % 5 == 0:
                env.render()

        # Resultados del episodio
        episode_result = "ÉXITO" if not truncated else "TIMEOUT" if steps >= env.max_steps else "FALLO"
        print(f"{episode_result} - Steps: {steps}, Reward total: {total_reward:.2f}")

        if episode_result == "ÉXITO":
            successes += 1
        
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_poses.append(episode_poses)
        all_actions.append(episode_actions)
        all_errors.append(episode_errors)

    return {
        'controller_type': controller_type,
        'rewards': all_rewards,
        'steps': all_steps,
        'poses': all_poses,
        'actions': all_actions,
        'errors': all_errors,
        'success_rate': successes / num_episodes
    }


def calculate_jerk(actions, dt):
    steering_angles = np.array(actions)[:, 1]  # Extraer solo los ángulos de dirección
    if len(steering_angles) < 2:
        return np.zeros_like(steering_angles)
    
    # Primera derivada (velocidad angular)
    angular_velocity = np.gradient(steering_angles, dt)
    
    # Segunda derivada (jerk angular)
    jerk = np.gradient(angular_velocity, dt)
    
    return jerk


# Genera gráficas comparativas entre PID y RL
def plot_comparison(pid_results, rl_results, reference_trajectory):

    # Primera figura: Comparativas básicas
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Extraer datos del primer episodio para comparar
    pid_poses = np.array(pid_results['poses'][0])
    rl_poses = np.array(rl_results['poses'][0])
    pid_errors = np.array(pid_results['errors'][0])
    rl_errors = np.array(rl_results['errors'][0])
    pid_actions = np.array(pid_results['actions'][0])
    rl_actions = np.array(rl_results['actions'][0])
    
    # 1. Trayectoria seguida
    gs = axes[0, 0].get_gridspec()
    # Eliminar los dos primeros ejes para fusionarlos
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.remove()
    ax_traj = fig.add_subplot(gs[0, :])

    ax_traj.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], 
                 'g--', linewidth=4, label='Referencia', alpha=0.5)
    ax_traj.plot(pid_poses[:, 0], pid_poses[:, 1], 
                 'b-', linewidth=1, label='PID')
    ax_traj.plot(rl_poses[:, 0], rl_poses[:, 1], 
                 'r-', linewidth=1, label='RL')

    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('Trayectoria Seguida')
    ax_traj.legend()
    ax_traj.grid(True)
    
    # 2. Error lateral comparativo
    axes[1, 0].plot(pid_errors[:, 0], 'b-', label='PID', alpha=0.8)
    axes[1, 0].plot(rl_errors[:, 0], 'r-', label='RL', alpha=0.8)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error Lateral (m)')
    axes[1, 0].set_title('Error Lateral Comparativo')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 3. Error de orientación comparativo
    axes[1, 1].plot(pid_errors[:, 1], 'b-', label='PID', alpha=0.8)
    axes[1, 1].plot(rl_errors[:, 1], 'r-', label='RL', alpha=0.8)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Error Orientación (rad)')
    axes[1, 1].set_title('Error de Orientación Comparativo')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 4. Ángulo de dirección
    pid_actions = np.array(pid_results['actions'][0])
    rl_actions = np.array(rl_results['actions'][0])
    axes[2, 0].plot(pid_actions[:, 1], 'b-', label='PID', alpha=0.8)
    axes[2, 0].plot(rl_actions[:, 1], 'r-', label='RL', alpha=0.8)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Ángulo de Dirección (rad)')
    axes[2, 0].set_title('Acciones de Control - Dirección')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # 5. Velocidad
    axes[2, 1].plot(pid_actions[:, 0], 'b-', label='PID', alpha=0.8)
    axes[2, 1].plot(rl_actions[:, 0], 'r-', label='RL', alpha=0.8)
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Velocidad (m/s)')
    axes[2, 1].set_title('Acciones de Control - Velocidad')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('images/comparison_pid_vs_rl.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # Segunda figura: Métricas avanzadas (IAE y Jerk)
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Error Cuadrático Medio Integrado (IAE) - Error Lateral
    iae_pid_lateral = np.cumsum(np.abs(pid_errors[:, 0]))
    iae_rl_lateral = np.cumsum(np.abs(rl_errors[:, 0]))
    
    axes[0, 0].plot(iae_pid_lateral, 'b-', label='PID', linewidth=2)
    axes[0, 0].plot(iae_rl_lateral, 'r-', label='RL', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('IAE Lateral (m)')
    axes[0, 0].set_title('Error Absoluto Integrado (IAE) - Error Lateral')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Error Cuadrático Medio Integrado (IAE) - Error Orientación
    iae_pid_orientation = np.cumsum(np.abs(pid_errors[:, 1]))
    iae_rl_orientation = np.cumsum(np.abs(rl_errors[:, 1]))
    
    axes[0, 1].plot(iae_pid_orientation, 'b-', label='PID', linewidth=2)
    axes[0, 1].plot(iae_rl_orientation, 'r-', label='RL', linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('IAE Orientación (rad)')
    axes[0, 1].set_title('Error Absoluto Integrado (IAE) - Error Orientación')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Análisis de Suavidad (Jerk) - Dirección
    # Jerk = derivada del ángulo de dirección (cambio brusco en el steering)
    dt = config.dt  # Intervalo de tiempo entre steps
    
    # Calcular jerk para PID
    pid_steering = pid_actions[:, 1]
    pid_steering_rate = np.diff(pid_steering) / dt  # Velocidad de cambio
    pid_jerk = np.diff(pid_steering_rate) / dt  # Aceleración de cambio (Jerk)
    
    # Calcular jerk para RL
    rl_steering = rl_actions[:, 1]
    rl_steering_rate = np.diff(rl_steering) / dt
    rl_jerk = np.diff(rl_steering_rate) / dt
    
    # Unir los dos subplots inferiores para el análisis de jerk
    gs = axes[1, 0].get_gridspec()
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.remove()
    ax_jerk = fig2.add_subplot(gs[1, :])

    ax_jerk.plot(range(len(pid_jerk)), np.abs(pid_jerk), 'b-', label='PID', alpha=0.8)
    ax_jerk.plot(range(len(rl_jerk)), np.abs(rl_jerk), 'r-', label='RL', alpha=0.8)
    ax_jerk.set_xlabel('Step')
    ax_jerk.set_ylabel('|Jerk| (rad/s³)')
    ax_jerk.set_title('Análisis de Suavidad - Jerk en Dirección')
    ax_jerk.legend()
    ax_jerk.grid(True)
    ax_jerk.set_yscale('log')  # Escala logarítmica para mejor visualización
        
    plt.tight_layout()
    plt.savefig('images/advanced_metrics_iae_jerk.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    # Estadísticas numéricas
    print("\n" + "="*50)
    print("COMPARATIVA ESTADÍSTICA")
    print("="*50)
    print(f"PID - Reward promedio: {np.mean(pid_results['rewards']):.2f} ± {np.std(pid_results['rewards']):.2f}")
    print(f"RL  - Reward promedio: {np.mean(rl_results['rewards']):.2f} ± {np.std(rl_results['rewards']):.2f}")
    print(f"PID - Steps promedio: {np.mean(pid_results['steps']):.1f} ± {np.std(pid_results['steps']):.1f}")
    print(f"RL  - Steps promedio: {np.mean(rl_results['steps']):.1f} ± {np.std(rl_results['steps']):.1f}")
    print(f"PID - Tasa de éxito: {pid_results['success_rate']*100:.1f}%")
    print(f"RL  - Tasa de éxito: {rl_results['success_rate']*100:.1f}%")
    print(f"PID - Error lateral medio: {np.mean(np.abs(pid_errors[:, 0])):.3f} m")
    print(f"RL  - Error lateral medio: {np.mean(np.abs(rl_errors[:, 0])):.3f} m")



if __name__ == "__main__":
    
    # Configuración
    MODEL_PATH = "controller/ackermann_ppo_model.zip"
    NUM_EPISODES = 1
    RENDER = False

    print("Cargando controladores...")

    # Controlador RL
    try:
        rl_controller = PPO.load(MODEL_PATH)
        print("Modelo RL cargado")
    except Exception as e:
        print(f"Error cargando modelo RL: {e}")
        exit()

    # Controlador PID (usa tus parámetros)
    pid_controller = PIDController(config.kp, config.ki, config.kd)
    print("Controlador PID configurado")

    # Crear entorno
    env = AckermannTrakingEnv(compare=True)
    reference_trajectory = env.reference_trajectory

    # Evaluar ambos controladores
    print("\nEvaluando PID...")
    pid_results = evaluate_controller(env, "PID", pid_controller, NUM_EPISODES, RENDER)

    print("\nEvaluando RL...")
    rl_results = evaluate_controller(env, "RL", rl_controller, NUM_EPISODES, RENDER)

    # Comparar y graficar resultados
    print("\nGenerando comparativa...")
    plot_comparison(pid_results, rl_results, reference_trajectory)

    print("Comparación completada. Resultados guardados en 'comparison_results.npz'")
    env.close()