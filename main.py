import os
import time
import numpy as np
from src.environments import get_env
from src.qlearning import train_q_learning
from src.dqn import train_dqn
from src.maskable_ppo import train_maskable_ppo
from src.masking import FrozenLakeMaskWrapper
from src.plotting import plot_multiple_agents
from src.simulate import simulate_agent
from src.policy import display_policy

CASE = "3"
AGENTS_TO_TEST = ["Q-Learning", "DQN", "MaskablePPO"]
N_RUNS = 5

Q_PARAMS = {
    "max_episodes": 3000, "gamma": 0.99, "alpha_init": 0.5, "alpha_min": 0.01,
    "alpha_decay": 0.995, "epsilon_init": 1.0, "epsilon_min": 0.05, 
    "epsilon_decay": 0.995, "print_freq": 500
}

DQN_PARAMS = {
    "learning_rate": 0.0005, "buffer_size": 50000, "learning_starts": 1000,
    "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.6,
    "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
    "target_update_interval": 1000, 
    "total_timesteps": 300000 
}

MASKABLE_PPO_PARAMS = {
    "learning_rate": 0.0005, "n_steps": 2048, "batch_size": 256,
    "gamma": 0.99, "ent_coef": 0.05, "clip_range": 0.2,
    "total_timesteps": 300000
}

def main():
    agents_data = {}
    best_models = {}

    total_time = 0
    total_convergence_rate = 0

    for agent in AGENTS_TO_TEST:
        print(f"\n{'='*40}\nCas d'étude : {CASE} | Agent : {agent} ({N_RUNS} runs)\n{'='*40}")
        
        all_rewards = []
        all_lengths = []
        
        q_kwargs = Q_PARAMS.copy()
        dqn_kwargs = DQN_PARAMS.copy()
        ppo_kwargs = MASKABLE_PPO_PARAMS.copy()

        start_time = time.time()

        for run in range(N_RUNS):
            print(f"Essai {run + 1}/{N_RUNS}")
            
            raw_env = get_env(CASE)
            if agent == "MaskablePPO":
                env = FrozenLakeMaskWrapper(raw_env)
            else:
                env = raw_env
            
            if agent == "Q-Learning":
                model, rew_hist, len_hist = train_q_learning(env, **q_kwargs)
                if run == 0: 
                    best_models[agent] = model
                    
            elif agent == "DQN":
                timesteps = dqn_kwargs.pop("total_timesteps")
                model, rew_hist, len_hist = train_dqn(env, total_timesteps=timesteps, **dqn_kwargs)
                dqn_kwargs["total_timesteps"] = timesteps
                if run == 0:
                    best_models[agent] = model

            elif agent == "MaskablePPO":
                timesteps = ppo_kwargs.pop("total_timesteps")
                model, rew_hist, len_hist = train_maskable_ppo(env, total_timesteps=timesteps, **ppo_kwargs)
                ppo_kwargs["total_timesteps"] = timesteps
                if run == 0:
                    best_models[agent] = model

            end_time = time.time()

            run_time = end_time - start_time
            run_convergence_rate = np.mean(len_hist[-100:]) if len(len_hist) > 0 else 0
            
            print(f"Temps d'exécution : {run_time:.2f} s")
            print(f"Taux de convergence : ~{int(run_convergence_rate)} pas")

            total_time += run_time
            total_convergence_rate += run_convergence_rate

            all_rewards.append(rew_hist)
            all_lengths.append(len_hist)
            env.close()

        print(f"\n{'='*40}\nRÉSUMÉ DES MÉTRIQUES ({N_RUNS} runs)\n{'='*40}")
        print(f"Temps d'execution (Moyen) : {total_time / N_RUNS:.2f} s")
        print(f"Taux de convergence (Pas) : ~{int(total_convergence_rate / N_RUNS)} pas par épisode en fin d'entraînement")
            
        agents_data[agent] = (all_rewards, all_lengths)

    print("\nGénération des graphiques comparatifs.")
    rewards_only_data = {k: v[0] for k, v in agents_data.items()}
    plot_multiple_agents(rewards_only_data, CASE)

    print("\nGénération des cartes de politiques et simulations.")
    for agent, model in best_models.items():
        if agent in ["Q-Learning", "Lagrangian-Q", "MaskablePPO"]:
            raw_eval_env = get_env(CASE)
            eval_env = FrozenLakeMaskWrapper(raw_eval_env) if agent == "MaskablePPO" else raw_eval_env
            display_policy(eval_env, model, agent, CASE)
            eval_env.close()

if __name__ == "__main__":
    main()