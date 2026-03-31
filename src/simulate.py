import time
import numpy as np

def simulate_agent(env, model, agent_name, num_episodes=3):
    """
    Simule et affiche le comportement de l'agent entraîné.
    """
    print(f"\nVisualisation de l'agent {agent_name}.")
    
    for _ in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        
        while not done:
            if agent_name == "Q-Learning":
                max_q_value = np.max(model[state])
                best_actions = np.where(model[state] == max_q_value)[0]
                action = np.random.choice(best_actions)
                
            elif agent_name == "DQN":
                action, _ = model.predict(state, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()
                    
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            time.sleep(0.1) 