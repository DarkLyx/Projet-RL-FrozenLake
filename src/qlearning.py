import numpy as np

def train_q_learning(
    env, 
    max_episodes=10000, 
    gamma=0.99,
    alpha_init=0.1, 
    alpha_min=0.01, 
    alpha_decay=0.9995,
    epsilon_init=1.0, 
    epsilon_min=0.05, 
    epsilon_decay=0.9995,
    print_freq=1000
):
    """
    Entraîne un agent Q-Learning.
    Tous les hyperparamètres sont configurables.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    q_table = np.zeros((n_states, n_actions)) 
    
    rewards_history = [] 
    lengths_history = []

    for episode in range(1, max_episodes + 1):
        alpha = max(alpha_min, alpha_init * (alpha_decay ** episode))
        epsilon = max(epsilon_min, epsilon_init * (epsilon_decay ** episode))

        state, _ = env.reset()
        terminal_state = False
        episode_reward = 0
        steps = 0

        while not terminal_state:
            steps += 1

            if np.random.rand() < epsilon:
                action = env.action_space.sample() 
            else:
                max_q_value = np.max(q_table[state])
                best_actions = np.where(q_table[state] == max_q_value)[0]
                action = np.random.choice(best_actions)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            terminal_state = terminated or truncated

            if terminated:
                max_q_next = 0.0 
            else:
                max_q_next = np.max(q_table[next_state])
                
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max_q_next - q_table[state, action])
            
            state = next_state
            episode_reward += reward

        rewards_history.append(1 if episode_reward > 0 else 0)
        lengths_history.append(steps)
        
        if episode % print_freq == 0:
            success_rate = np.mean(rewards_history[-print_freq:]) * 100
            print(f"Épisode : {episode:5d} | eps : {epsilon:.3f} | alpha : {alpha:.3f} | Succès ({print_freq} derniers) : {success_rate:.1f}%")
        
    return q_table, rewards_history, lengths_history