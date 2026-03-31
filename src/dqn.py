from stable_baselines3 import DQN
from src.metrics import SB3MetricsCallback

def train_dqn(env, total_timesteps=100000, **dqn_kwargs):
    """
    Entraîne un agent DQN avec StableBaselines3.
    Tous les hyperparamètres sont configurables.
    """
    callback = SB3MetricsCallback()
    
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model, callback.rewards_history, callback.lengths_history