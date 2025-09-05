import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class TrackingCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=0):
        super(TrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Acumular recompensa del episodio actual
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Verificar si el episodio terminó
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        # Log cada check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                # Media de últimos 20 episodios
                mean_reward = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths[-20:]) if len(self.episode_lengths) >= 20 else np.mean(self.episode_lengths)
                
                self.mean_rewards.append(mean_reward)

                print(f"Step {self.n_calls}")
                print(f"  Mean Episode Reward (last 20): {mean_reward:.2f}")
                print(f"  Mean Episode Length (last 20): {mean_length:.1f}")
                print(f"  Total Episodes: {len(self.episode_rewards)}")
                
                # Log para TensorBoard
                self.logger.record('train/mean_reward', mean_reward)
                self.logger.record('train/mean_episode_length', mean_length)
                self.logger.record('train/total_episodes', len(self.episode_rewards))
                
        
        return True
    
    def _on_training_end(self):
        # Guardar recompensas finales
        if len(self.episode_rewards) > 0:
            final_mean_reward = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else np.mean(self.episode_rewards)
            print(f"Training completed. Final mean reward: {final_mean_reward:.2f}")
            print(f"Total episodes: {len(self.episode_rewards)}")

            self._plot_learning_curve()


    # Genera una gráfica la curva de aprendizaje
    def _plot_learning_curve(self):

        plt.figure(figsize=(10, 6))
        
        # Gráfica de recompensas por episodio
        episodes = range(1, len(self.episode_rewards) + 1)
        plt.plot(episodes, self.episode_rewards, 'bo-', alpha=0.7, 
             markersize=4, linewidth=1, label='Recompensa por episodio')
        
        # Media móvil con ventana más grande (20-30% del total de episodios)
        window_size = max(5, min(20, len(self.episode_rewards) // 3))  # Ajuste automático
        # if window_size > 0:
        #     moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
        #     plt.plot(range(window_size, len(self.episode_rewards) + 1), moving_avg, 
        #             'r-', linewidth=3, label=f'Media móvil ({window_size} episodios)')
        
        plt.xlabel('Episodio de Entrenamiento')
        plt.ylabel('Recompensa Total')
        plt.title('Curva de Aprendizaje - Evolución de la Recompensa')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir texto con estadísticas finales
        final_stats = (f'Episodios: {len(self.episode_rewards)}\n'
                      f'Recompensa máxima: {np.max(self.episode_rewards):.1f}\n'
                      f'Recompensa media: {np.mean(self.episode_rewards):.1f}')
        plt.text(0.02, 0.98, final_stats, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('./images/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.savefig('./images/learning_curve.eps', format='eps', dpi=1000)
        plt.show()



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
    
