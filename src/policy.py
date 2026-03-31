import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def display_policy(env, model, agent_name, case_name):
    """
    Génère et sauvegarde une image de la politique apprise avec Matplotlib.
    """
    action_mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    desc = env.unwrapped.desc.astype(str)
    size = desc.shape[0]
    
    color_matrix = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            if desc[r, c] == 'S': color_matrix[r, c] = 1
            elif desc[r, c] == 'H': color_matrix[r, c] = 2
            elif desc[r, c] == 'G': color_matrix[r, c] = 3
            
    cmap = ListedColormap(['#f0f0f0', "#0000FF", "#000000", "#FF0000"])
    
    fig, ax = plt.subplots(figsize=(size * 0.8, size * 0.8))
    ax.matshow(color_matrix, cmap=cmap)
    
    for r in range(size):
        for c in range(size):
            state = r * size + c
            cell_type = desc[r, c]
            
            if cell_type in ['H', 'G', 'S']:
                text_color = 'white' if cell_type in ['H', 'S'] else 'black'
                ax.text(c, r, cell_type, va='center', ha='center', 
                        weight='bold', fontsize=14, color=text_color)
            else:
                if agent_name == "Q-Learning":
                    action = np.argmax(model[state])
                elif agent_name == "DQN":
                    action, _ = model.predict(state, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()
                
                arrow = action_mapping[action]
                ax.text(c, r, arrow, va='center', ha='center', 
                        fontsize=16, color='#333333')
                
    ax.set_xticks(np.arange(-.5, size, 1))
    ax.set_yticks(np.arange(-.5, size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='white', linestyle='-', linewidth=2)
    ax.tick_params(axis='both', which='both', length=0) # Cache les petits traits
    
    plt.title(f"Politique apprise - {agent_name} (Cas {case_name})", pad=20)
    
    os.makedirs("results", exist_ok=True)
    filepath = f"results/Policy_{case_name}_{agent_name}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Carte des politiques sauvegardée : {filepath}")