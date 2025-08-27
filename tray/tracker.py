import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrackingCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=0):
        super(TrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Acumular recompensa del episodio actual
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        
        # Verificar si el episodio terminó
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
        # Log cada check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])  # Media de últimos 10 episodios
                print(f"Step {self.n_calls}, Mean Episode Reward: {mean_reward:.2f}")
                # También puedes guardar esto para TensorBoard
                self.logger.record('train/mean_reward', mean_reward)
        
        return True
    
    def _on_training_end(self):
        # Guardar recompensas finales
        if len(self.episode_rewards) > 0:
            final_mean_reward = np.mean(self.episode_rewards[-10:])
            print(f"Training completed. Final mean reward: {final_mean_reward:.2f}")



# Evaluación completa con métricas y gráficas
def extract_metrics(env, episode, save_plots=True):

    all_metrics = []
    
    print(f"\n=== EPISODIO {episode + 1} ===")
        
    # Obtener métricas del episodio
    metrics = env.get_metrics()
    all_metrics.append(metrics)
        
    print(f"Reward total: {metrics['total_reward']:.1f}")
    print(f"Velocidad media: {metrics['mean_speed']:.2f} m/s")
    print(f"Error lateral medio: {metrics['mean_lateral_error']:.2f} m")
    print(f"Pasos: {metrics['episode_length']}")
        
    # Guardar gráficas
    if save_plots:
        env.plot_metrics(save_path=f"./images/episode_{episode+1}_metrics.png")

    # return all_metrics
    
