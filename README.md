# Control Predictivo USV

Control Predictivo basado en Aprendizaje por Refuerzo para el seguimiento de trayectorias de un vehículo autónomo (Ackermann) para la asignatura Control Inteligente del Máster en Ingeniería de Sistemas y Control (UNED).

## Características

- **Entorno Gym personalizado** para simulación de seguimiento de trayectorias.
- **Controlador PID tradicional** y controlador basado en **Aprendizaje por Refuerzo (RL, PPO)**.
- **Comparativa automática** entre PID y RL con métricas y gráficas.
- **Visualización** de trayectorias, errores y recompensas.

## Estructura

- `gym/env.py`: Entorno de simulación.
- `controller/pid_controller.py`: Controlador PID.
- `main_train.py`: Entrenamiento del modelo RL.
- `main.py`: Evaluación del modelo RL.
- `main_pid.py`: Evaluación del PID.
- `main_compare.py`: Comparativa PID vs RL.
- `tray/tracker.py`: Métricas y gráficas.
- `config.py`: Parámetros de simulación y control.

## Uso rápido

1. **Entrenar RL**:
   ```bash
   python main_train.py
   ```
2. **Evaluar RL**:
   ```bash
   python main.py
   ```
3. **Evaluar PID**:
   ```bash
   python main_pid.py
   ```
4. **Comparar PID vs RL**:
   ```bash
   python main_compare.py
   ```

Las gráficas y resultados se guardan en la carpeta `images/`.

## Requisitos

- Python 3.8+
- stable-baselines3
- gymnasium
- matplotlib
- numpy


## Contacto

Ante cualquier duda consultar a través de los medios indicados en el perfil.
