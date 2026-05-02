from sb3_contrib import MaskablePPO
from src.metrics import SB3MetricsCallback

def train_maskable_ppo(env, total_timesteps=100000, **ppo_kwargs):
    callback = SB3MetricsCallback()
    
    model = MaskablePPO("MlpPolicy", env, verbose=0, **ppo_kwargs)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model, callback.rewards_history, callback.lengths_history