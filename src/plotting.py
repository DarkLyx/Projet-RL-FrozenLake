import matplotlib.pyplot as plt
import numpy as np
import os

def plot_multiple_agents(agents_data, case_name, window_size=100):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (agent_name, all_runs_history), color in zip(agents_data.items(), colors):
        smoothed_runs = []
        for history in all_runs_history:
            if len(history) >= window_size:
                smoothed = np.convolve(history, np.ones(window_size)/window_size, mode='valid') * 100
                smoothed_runs.append(smoothed)
                
        if smoothed_runs:
            min_len = min(len(run) for run in smoothed_runs)
            smoothed_runs = [run[:min_len] for run in smoothed_runs]
            smoothed_runs = np.array(smoothed_runs)
            mean_curve = np.mean(smoothed_runs, axis=0)
            std_curve = np.std(smoothed_runs, axis=0)
            
            x = np.arange(len(mean_curve))
            plt.plot(x, mean_curve, label=f"{agent_name} ({len(all_runs_history)} runs)", linewidth=2, color=color)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3, color=color)
            
    plt.title(f"Comparaison des stratégies - Cas d'étude {case_name}")
    plt.xlabel(f"Épisodes (Lissés sur {window_size})")
    plt.ylabel("Succès (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    
    filepath = f"results/Comparaison_Cas_{case_name}.png"
    plt.savefig(filepath, dpi=300)
    plt.close()