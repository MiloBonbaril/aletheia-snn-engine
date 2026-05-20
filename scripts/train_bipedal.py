#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import gymnasium as gym

# Reconfigure stdout/stderr to use UTF-8 on Windows command prompts to prevent UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure we can import the compiled python_bridge from the virtual environment
try:
    import python_bridge
    if not hasattr(python_bridge, "FastBrain"):
        raise ImportError("python_bridge was imported but does not contain FastBrain. It might be a namespace package.")
except ImportError:
    # Fallback to check parent directories if needed
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        import python_bridge
        if not hasattr(python_bridge, "FastBrain"):
            raise ImportError("python_bridge was imported from parent but does not contain FastBrain.")
    except ImportError as e:
        print("[ERROR] python_bridge is not installed or compiled. Please run 'maturin develop' first.")
        print(e)
        sys.exit(1)

def evaluate_brain(brain, env, max_steps):
    """
    Evaluates a single SNN brain clone in the Gymnasium environment.
    """
    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while True:
        # 1. Ingest observation states
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
        
        # 2. SNN propagation inside the high-performance Rust FFI
        actions = brain.tick(obs_list)
        
        # 3. Step the Box2D physics environment
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated or step_count >= max_steps:
            break
            
    return total_reward, step_count

def run_evolutionary_training(generations=100, max_steps=500, render=False, population_size=20, mutation_rate=0.15, mutation_strength=0.1):
    print("=" * 70)
    print("[ALETHEIA SNN ENGINE] - EVOLUTIONARY STRATEGY SNN TRAINING")
    print("=" * 70)
    print(f"Hyperparameters:")
    print(f"  - Generations: {generations}")
    print(f"  - Population Size: {population_size} clones/gen")
    print(f"  - Mutation Rate: {mutation_rate * 100:.1f}%")
    print(f"  - Mutation Strength: {mutation_strength:.2f}")
    print(f"  - Max Steps per Episode: {max_steps}")
    print(f"  - Render Mode: {'Human GUI Showcase' if render else 'High-Speed Headless'}")
    print("-" * 70)

    # Initialize the non-rendered training environment
    try:
        train_env = gym.make("BipedalWalker-v3")
    except Exception as e:
        print(f"[ERROR] Could not create Gymnasium environment: {e}")
        print("Please ensure gymnasium[box2d] is correctly installed.")
        sys.exit(1)

    print("FastBrain FFI bridge ready. Initializing SNN champion...")
    # Initialize the SNN champion brain (24 inputs, 36 hidden, 4 outputs)
    champion = python_bridge.FastBrain()
    
    # Evaluate the initial, unmutated SNN champion
    champion_fitness, initial_steps = evaluate_brain(champion, train_env, max_steps)
    print(f"Initial SNN Champion Fitness Score: {champion_fitness:.2f} (Steps: {initial_steps})")
    print("Starting parallelized evolutionary training loop...")
    print("-" * 70)

    fitness_history = [champion_fitness]
    start_time = time.time()

    for gen in range(1, generations + 1):
        gen_start = time.time()
        
        # 1. Selection & Mutation: Generate population pool of mutated clones using Rust FFI
        # This operates at memory speed without Python GIL or heap allocation overhead
        clones = [champion.mutate(mutation_rate, mutation_strength) for _ in range(population_size)]
        
        fitness_scores = []
        total_steps = 0
        
        # 2. Evaluation: Evaluate all offspring clones in sequential episodes
        for i, clone in enumerate(clones):
            score, steps = evaluate_brain(clone, train_env, max_steps)
            fitness_scores.append((clone, score))
            total_steps += steps

        # Find the best performing clone in this generation
        best_clone, best_fitness = max(fitness_scores, key=lambda x: x[1])
        avg_fitness = np.mean([s[1] for s in fitness_scores])
        
        gen_duration = time.time() - gen_start
        steps_per_sec = total_steps / gen_duration if gen_duration > 0 else 0

        # 3. Replacement / Parent Promotion Logic
        promoted = False
        if best_fitness > champion_fitness:
            # Stage the mutated core weights into the champion's ArcSwap slot atomically
            champion.swap_brain(best_clone)
            
            # The next tick() call in the champion SNN will automatically trigger the lock-free swap
            # We also update our reference fitness
            champion_fitness = best_fitness
            promoted = True
            
        fitness_history.append(champion_fitness)

        # 4. Stylized Console Reporting
        promotion_marker = " [PROMOTED CHAMPION!]" if promoted else ""
        print(f"[Gen {gen:3d}/{generations:3d}] Best: {best_fitness:7.2f} | Avg: {avg_fitness:7.2f} | Champion: {champion_fitness:7.2f} | Speed: {steps_per_sec:6.1f} steps/s{promotion_marker}")

        # 5. Live Showcase Human rendering
        if render and (gen % 5 == 0 or promoted):
            print(f"--> Running Showcase Render for Champion (Gen {gen}, Fitness: {champion_fitness:.2f})")
            try:
                render_env = gym.make("BipedalWalker-v3", render_mode="human")
                evaluate_brain(champion, render_env, max_steps)
                render_env.close()
            except Exception as e:
                print(f"[WARNING] Could not run render showcase: {e}")

    total_duration = time.time() - start_time
    train_env.close()
    
    print("=" * 70)
    print("EVOLUTIONARY TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total training time: {total_duration / 60.0:.2f} minutes")
    print(f"Initial Champion Fitness: {fitness_history[0]:.2f}")
    print(f"Final Champion Fitness: {champion_fitness:.2f}")
    print(f"Net Fitness Improvement: {champion_fitness - fitness_history[0]:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aletheia SNN Engine - Evolutionary Walker Trainer")
    parser.add_argument("-g", "--generations", type=int, default=50, help="Number of evolutionary generations to train")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("-r", "--render", action="store_true", help="Enable visual showcase rendering")
    parser.add_argument("-p", "--population", type=int, default=20, help="Population size (mutated SNN clones per generation)")
    parser.add_argument("-m", "--mutation-rate", type=float, default=0.15, help="Probability of mutating a synapse weight (0.0 to 1.0)")
    parser.add_argument("--mutation-strength", type=float, default=0.1, help="Max perturbation adjustment to synapse weights")
    
    args = parser.parse_args()
    
    run_evolutionary_training(
        generations=args.generations,
        max_steps=args.steps,
        render=args.render,
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength
    )
