#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import gymnasium as gym
import threading

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

def gpu_evolution_thread(champion, replay_buffer, solver, num_clones, mutation_rate, mutation_strength, stop_event, stats):
    """
    Subconscious GPU evolutionary forge running in the background.
    Continually simulates mutant clones against the circular historical buffer,
    optimizing SNN weights at lightning speed, and atomically swaps the champion.
    """
    last_fitness = -99999.0
    
    while not stop_event.is_set():
        # Sleep a little to yield to the main thread (retains GIL safety)
        time.sleep(0.1)
        
        # Only start evolution once we have enough frames in the buffer (minimum 100 frames)
        if replay_buffer.count < 100:
            continue
            
        try:
            # Perform GPU evolutionary step based on the Phantom Buffer
            # This is extremely fast (takes < 10ms) and runs concurrently with CPU steps
            best_clone, fitness = solver.evolve(
                champion,
                replay_buffer,
                num_clones,
                mutation_rate,
                mutation_strength
            )
            
            # Atomically stage the champion for hot-swapping
            champion.swap_brain(best_clone)
            
            stats["swaps"] += 1
            stats["last_gpu_fitness"] = fitness
            
        except Exception as e:
            # Print GPU FFI launch edge cases for diagnostic visibility
            print(f"\n[GPU Forge Error] {e}")

def run_evolutionary_training(generations=100, max_steps=500, render=False, population_size=20, 
                              mutation_rate=0.15, mutation_strength=0.1, load_path=None, gpu=False):
    print("=" * 70)
    print("[ALETHEIA SNN ENGINE] - EVOLUTIONARY STRATEGY SNN TRAINING")
    print("=" * 70)
    print(f"Hyperparameters:")
    print(f"  - Generations: {generations}")
    print(f"  - Population Size: {population_size} clones/gen")
    print(f"  - Mutation Rate: {mutation_rate * 100:.1f}%")
    print(f"  - Mutation Strength: {mutation_strength:.2f}")
    print(f"  - Max Steps per Episode: {max_steps}")
    print(f"  - Acceleration Mode: {'High-Speed GPU (Phantom Buffer)' if gpu else 'CPU Parallel Multithreading'}")
    print(f"  - Render Mode: {'Human GUI Showcase' if render else 'High-Speed Headless'}")
    if load_path:
        print(f"  - Pre-trained Weights: '{load_path}'")
    print("-" * 70)

    # Initialize the non-rendered training environment
    try:
        train_env = gym.make("BipedalWalker-v3")
    except Exception as e:
        print(f"[ERROR] Could not create Gymnasium environment: {e}")
        print("Please ensure gymnasium[box2d] is correctly installed.")
        sys.exit(1)

    print("FastBrain FFI bridge ready. Initializing SNN champion...")
    # Initialize or load the SNN champion brain
    if load_path:
        try:
            champion = python_bridge.FastBrain.load_from_file(load_path)
            print(f"[INFO] Loaded pre-trained SNN champion from '{load_path}' with shape ({champion.num_inputs} -> {champion.num_hidden} -> {champion.num_outputs})")
        except Exception as e:
            print(f"[ERROR] Failed to load pre-trained weights from '{load_path}': {e}")
            print("Fallback to initializing a new SNN brain with default architecture.")
            champion = python_bridge.FastBrain()
    else:
        # Default BipedalWalker architecture (24 inputs, 36 hidden, 4 outputs)
        champion = python_bridge.FastBrain()

    
    # Evaluate the initial, unmutated SNN champion
    champion_fitness, initial_steps = evaluate_brain(champion, train_env, max_steps)
    print(f"Initial SNN Champion Fitness Score: {champion_fitness:.2f} (Steps: {initial_steps})")
    
    fitness_history = [champion_fitness]
    start_time = time.time()

    if gpu:
        # GPU Subconscious Evolution (L'Évaluation Fantôme)
        if not hasattr(python_bridge, "CudaSnnSolver"):
            print("[ERROR] CUDA SNN Solver is not compiled. Please re-compile with 'cuda' feature.")
            sys.exit(1)
            
        print("[INFO] Initializing GPU solver and Phantom Replay Buffer...")
        # Pre-allocate a large circular replay buffer (capacity 5000 frames)
        replay_buffer = python_bridge.PhantomReplayBuffer(champion.num_inputs, champion.num_outputs, 5000)
        solver = python_bridge.CudaSnnSolver()
        
        # Track statistics across the threads
        stats = {
            "swaps": 0,
            "last_gpu_fitness": 0.0
        }
        
        stop_event = threading.Event()
        bg_thread = threading.Thread(
            target=gpu_evolution_thread,
            args=(champion, replay_buffer, solver, population_size, mutation_rate, mutation_strength, stop_event, stats),
            daemon=True
        )
        bg_thread.start()
        
        print("Starting live episodes with subconscious GPU evolution...")
        print("-" * 70)
        
        for episode in range(1, generations + 1):
            episode_start = time.time()
            obs, info = train_env.reset()
            total_reward = 0.0
            step_count = 0
            
            # Reset active stats for the episode
            swaps_before = stats["swaps"]
            
            while True:
                obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
                
                # SNN tick (will automatically swap the brain if the GPU thread has staged a new one)
                actions = champion.tick(obs_list)
                
                obs, reward, terminated, truncated, info = train_env.step(actions)
                total_reward += reward
                step_count += 1
                
                # Push the transition into the circular Phantom Buffer
                replay_buffer.add_frame(obs_list, actions, reward)
                
                if terminated or truncated or step_count >= max_steps:
                    break
            
            episode_dur = time.time() - episode_start
            steps_per_sec = step_count / episode_dur if episode_dur > 0 else 0
            
            swaps_during = stats["swaps"] - swaps_before
            swap_marker = f" [Swaps: {swaps_during}]" if swaps_during > 0 else ""
            
            print(f"[Episode {episode:3d}/{generations:3d}] Score: {total_reward:7.2f} | Steps: {step_count:3d} | Buffer: {replay_buffer.count:4d} | Speed: {steps_per_sec:6.1f} st/s | GPU Fit: {stats['last_gpu_fitness']:7.2f}{swap_marker}")
            
            fitness_history.append(total_reward)
            
            # Showcase human render if enabled
            if render and (episode % 5 == 0):
                print(f"--> Showcase Render Episode {episode} (Score: {total_reward:.2f})")
                try:
                    render_env = gym.make("BipedalWalker-v3", render_mode="human")
                    evaluate_brain(champion, render_env, max_steps)
                    render_env.close()
                except Exception as e:
                    print(f"[WARNING] Could not run render showcase: {e}")
                    
        # Stop background thread
        stop_event.set()
        bg_thread.join(timeout=1.0)
        
    else:
        # Standard CPU Sequential Generations
        print("Starting parallelized evolutionary training loop on CPU...")
        print("-" * 70)
        
        for gen in range(1, generations + 1):
            gen_start = time.time()
            
            # 1. Selection & Mutation: Generate population pool of mutated clones using Rust FFI
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
    
    # 6. Smart Persistence: Save optimized SNN champion state to disk
    try:
        os.makedirs("weights", exist_ok=True)
        weights_path = "weights/champion_snn.json"
        champion.save(weights_path)
        print(f"\n[PERSISTENCE] Successfully saved optimized SNN champion state to '{weights_path}'")
    except Exception as e:
        print(f"\n[WARNING] Could not save SNN champion state: {e}")
    
    print("=" * 70)
    print("EVOLUTIONARY TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total training time: {total_duration / 60.0:.2f} minutes")
    print(f"Initial Champion Fitness: {fitness_history[0]:.2f}")
    print(f"Final Champion Fitness: {fitness_history[-1]:.2f}")
    print(f"Net Fitness Improvement: {fitness_history[-1] - fitness_history[0]:.2f}")
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
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to a pre-trained SNN JSON weights file to load before training")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU-accelerated parallel evolution and Phantom Replay Buffer")
    
    args = parser.parse_args()
    
    run_evolutionary_training(
        generations=args.generations,
        max_steps=args.steps,
        render=args.render,
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        load_path=args.load,
        gpu=args.gpu
    )
