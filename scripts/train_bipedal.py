#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import gymnasium as gym

# Ensure we can import the compiled python_bridge from the virtual environment
try:
    import python_bridge
except ImportError:
    # Fallback to check parent directories if needed
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        import python_bridge
    except ImportError as e:
        print("❌ Error: python_bridge is not installed or compiled. Please run 'maturin develop' first.")
        print(e)
        sys.exit(1)

def run_simulation(episodes=5, max_steps=500, render=False):
    print("=" * 60)
    print("🧠 ALETHEIA SNN ENGINE - BIPEDAL WALKER TRAINING RUN 🧠")
    print("=" * 60)
    
    render_mode = "human" if render else None
    print(f"Initializing Gymnasium 'BipedalWalker-v3' (render_mode={render_mode})...")
    
    try:
        env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("Please ensure that gymnasium[box2d] is correctly installed.")
        sys.exit(1)
        
    print("FastBrain FFI bridge ready. Starting live spiking simulation...")
    print("-" * 60)
    
    # Initialize the spiking neural network through PyO3 bridge
    brain = python_bridge.FastBrain()
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        active_neurons_count = 0
        spikes_mask = 0
        start_time = time.time()
        
        while True:
            # 1. Python (Gymnasium) says: "Here is the state of BipedalWalker (24 floats)"
            obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
            
            # 2. PyO3 FFI passes observations to the core_engine in RAM and runs SNN inference
            actions = brain.tick(obs_list)
            
            # 3. FastBrain has pushed the active neurons bitmask to telemetry in RAM
            # Let's inspect the active neuron spikes mask
            spikes_mask = brain.get_last_spikes()
            active_neurons_count = bin(spikes_mask).count('1')
            
            # 4. Python retrieves the 4 actions and steps the environment
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            step_count += 1
            
            # If rendering in human mode, sleep briefly so a human can track the motion
            if render:
                time.sleep(1.0 / 60.0)
                
            # Periodic logging to show the real-time FFI cycle is alive and well
            if step_count % 100 == 0:
                print(f"[Ep {episode}] Step {step_count:4d} | Reward: {episode_reward:7.2f} | Active Spikes: {active_neurons_count:2d} | Bitmask: {hex(spikes_mask)}")
            
            if terminated or truncated or step_count >= max_steps:
                break
                
        duration = time.time() - start_time
        fps = step_count / duration if duration > 0 else 0
        print(f"✅ Episode {episode} Finished! Total Steps: {step_count:4d} | Reward: {episode_reward:7.2f} | FFI Speed: {fps:.1f} Hz | Last Active Spikes: {active_neurons_count}")
        print("-" * 60)
        
    env.close()
    print("=" * 60)
    print("🎉 BipedalWalker simulation complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aletheia SNN Engine - BipedalWalker Trainer")
    parser.add_argument("-e", "--episodes", type=int, default=5, help="Number of episodes to simulate")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("-r", "--render", action="store_true", help="Enable visual human render mode (debug/eval)")
    
    args = parser.parse_args()
    
    run_simulation(
        episodes=args.episodes,
        max_steps=args.steps,
        render=args.render
    )
