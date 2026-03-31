import os
import time
import numpy as np
from src.environments import get_env
from src.qlearning import train_q_learning
from src.dqn import train_dqn
from src.plotting import plot_confidence_interval
from src.simulate import simulate_agent
from src.policy import display_policy

CASE = "1"  # "1", "2" ou "3"
AGENT = "Q-Learning"  # "Q-Learning" ou "DQN"
N_RUNS = 5

Q_PARAMS = {
    "max_episodes": 5000, "gamma": 0.99, "alpha_init": 0.5, "alpha_min": 0.01,
    "alpha_decay": 0.99, "epsilon_init": 1.0, "epsilon_min": 0.05, 
    "epsilon_decay": 0.99, "print_freq": 1000
}

DQN_PARAMS = {
    "learning_rate": 0.0005, "buffer_size": 50000, "learning_starts": 1000,
    "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.6,
    "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
    "target_update_interval": 1000, "total_timesteps": 50000
}

def main():
    print(f"\n{'='*40}\nCas d'étude : {CASE} | Agent : {AGENT} ({N_RUNS} runs)\n{'='*40}")
    
    all_rewards = []
    all_lengths = []
    best_model = None

    total_time = 0
    total_convergence_rate = 0

    for run in range(N_RUNS):
        print(f"\nEssai {run + 1}/{N_RUNS}")
        env = get_env(CASE)

        start_time = time.time()
        
        if AGENT == "Q-Learning":
            model, rew_hist, len_hist = train_q_learning(env, **Q_PARAMS)
            if run == 0: 
                os.makedirs("tables", exist_ok=True)
                np.save(f"tables/q_table_{CASE}.npy", model)
                best_model = model
                
        elif AGENT == "DQN":
            timesteps = DQN_PARAMS.pop("total_timesteps")
            model, rew_hist, len_hist = train_dqn(env, total_timesteps=timesteps, **DQN_PARAMS)
            DQN_PARAMS["total_timesteps"] = timesteps
            
            if run == 0:
                os.makedirs("models", exist_ok=True)
                model.save(f"models/dqn_model_{CASE}")
                best_model = model
                
        end_time = time.time()

        run_time = end_time - start_time
        run_convergence_rate = np.mean(len_hist[-100:]) if len(len_hist) > 0 else 0
        
        print(f"Temps d'exécution : {run_time:.2f} s")
        print(f"Taux de convergence : ~{int(run_convergence_rate)} pas")

        total_time += run_time
        total_convergence_rate += run_convergence_rate
        all_rewards.append(rew_hist)

        env.close()

    print(f"\n{'='*40}\nRÉSUMÉ DES MÉTRIQUES ({N_RUNS} runs)\n{'='*40}")
    print(f"Temps d'execution (Moyen) : {total_time / N_RUNS:.2f} s")
    print(f"Taux de convergence (Pas) : ~{int(total_convergence_rate / N_RUNS)} pas par épisode en fin d'entraînement")
        
    print("\nGénération du graphique.")
    plot_confidence_interval(all_rewards, AGENT, CASE)

    print("\nGénération de la carte de politique")
    eval_env = get_env(CASE)
    display_policy(eval_env, best_model, AGENT, CASE)
    eval_env.close()
    
    print(f"\nSimulation de {AGENT} pour le cas d'étude {CASE}")
    simulation = get_env(CASE, render_mode="human")
    
    simulate_agent(simulation, best_model, AGENT) 
    
    simulation.close()

if __name__ == "__main__":
    main()