import gymnasium as gym

def get_env(case_name, render_mode=None):
    if case_name == "1":
        return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=render_mode)
    
    elif case_name == "2":
        return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode=render_mode)
    
    elif case_name == "3":
        custom_map = [
            "SFFFFFFF", 
            "FFFFFFFF", 
            "HHHHFFFF", 
            "FFFHFFFF",
            "FFFHFFFF", 
            "HHHHFFFF", 
            "FFFFFFFF", 
            "GFFFFFFF"
        ]
        return gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True, render_mode=render_mode)