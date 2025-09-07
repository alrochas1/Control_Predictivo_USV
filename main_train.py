# Este es para entrenar el modelo (guarda un archivo zip con el modelo)

import numpy as np

import config
from gym.env import AckermannTrakingEnv
from tray.tracker import TrackingCallback

import sys
import types

# Crea un módulo 'gym' falso si no existe (hace falta para guardar el modelo)
if "gym" not in sys.modules:
    gym_fake = types.ModuleType("gym")
    gym_fake.__version__ = "0.21.0"
    sys.modules["gym"] = gym_fake
else:
    sys.modules["gym"].__version__ = "0.21.0"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


env = AckermannTrakingEnv()
env = DummyVecEnv([lambda: env])  # Para paralelización

# Crear callback (para debug)
callback = TrackingCallback(check_freq=100)

# Configurar PPO con parámetros para control continuo
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,    # Learning rate más bajo para más estabilidad
    n_steps=2048,            # Pasos por episodio
    batch_size=64,           # Tamaño del batch
    n_epochs=10,             # Épocas de optimización por rollout
    gamma=0.99,              # Factor de descuento
    gae_lambda=0.95,         # Parámetro GAE
    clip_range=0.2,          # Clipping del ratio de políticas
    ent_coef=0.01,           # Coeficiente de entropía (exploración)
    vf_coef=0.5,             # Coeficiente value function
    max_grad_norm=0.5,       # Clip gradientes
    tensorboard_log="./tb_logs/"  # Logs
)

# Entrenar
print("Comenzando entrenamiento...")
try:
    model.learn(
        total_timesteps=200000,
        tb_log_name="PPO_ACKERMANN",
        callback=callback
    )
except KeyboardInterrupt:
    print("Entrenamiento interrumpido por el usuario")

# Guardar modelo
print("Guardando modelo...")
model.save("controller/ackermann_ppo_model")

# Cerrar entorno
env.close()
print("Entrenamiento completado. Modelo guardado como 'ackermann_ppo_model'")

