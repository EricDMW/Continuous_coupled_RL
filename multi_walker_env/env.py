import numpy as np
from pettingzoo.sisl import waterworld_v4

from matplotlib import pyplot as plt
import time

def main():
    # Create the WaterWorld environment with minimal parameters
    print("Creating WaterWorld environment...")
    
    # Try with only essential parameters
    env = waterworld_v4.parallel_env(
        render_mode="human",     # Set to "human" for visualization
        n_pursuers=5
    )
    
    print(f"Environment created with {env.max_num_agents} pursuers")
    
    # Start a new episode
    print("Starting new episode...")
    observations = env.reset()
    
    # Loop through the environment steps
    print("Running random actions...")
    episode_rewards = 0
    
    
    stepwise_reward = []
    for step in range(5000):  # Maximum 500 steps
        # Generate random actions for each agent
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Take a step in the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Calculate total reward for this step
        step_reward = sum(rewards.values())
        
        # Record current reward
        stepwise_reward.append(step_reward)
        
        episode_rewards += step_reward
        
        # if step % 50 == 0:  # Print status every 50 steps
        #     print(f"Step {step}: Total reward so far: {episode_rewards:.2f}")
        
        # # Brief pause to make visualization easier to follow
        # time.sleep(0.01)
        
        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()):
            print(f"Episode ended early at step {step}")
            break
    
    print(f"Episode complete. Total reward: {episode_rewards:.2f}")
    
    # Close the environment
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    main()