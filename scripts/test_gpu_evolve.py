#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Reconfigure stdout/stderr to use UTF-8 on Windows command prompts
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure we can import the compiled python_bridge
try:
    import python_bridge
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import python_bridge

def test_gpu_evolution():
    print("=" * 70)
    print("[ALETHEIA SNN] - INTEGRATION TEST FOR GPU EVOLUTION")
    print("=" * 70)

    # 1. Check feature availability
    if not hasattr(python_bridge, "CudaSnnSolver"):
        print("[SKIP] CudaSnnSolver is not available in the compiled binary (build without 'cuda' feature?).")
        sys.exit(1)

    print("[INFO] CUDA SNN Solver detected. Initializing component graph...")

    # 2. Instantiate network dimensions and agents
    num_inputs = 3
    num_hidden = 4
    num_outputs = 2
    capacity = 100
    num_clones = 64

    parent = python_bridge.FastBrain(num_inputs, num_hidden, num_outputs)
    buffer = python_bridge.PhantomReplayBuffer(num_inputs, num_outputs, capacity)
    solver = python_bridge.CudaSnnSolver()

    print(f"Parent Brain: {num_inputs} Inputs -> {num_hidden} Hidden -> {num_outputs} Outputs")
    print(f"Phantom Replay Buffer capacity: {capacity}")
    print(f"Solver active and bound to GPU.")

    # 3. Populate buffer with deterministic training trajectory
    print("[INFO] Populating replay buffer with simulated walking observations...")
    for i in range(capacity):
        # Generate stable patterns (e.g. sinusoidal gait simulation)
        angle = i * 0.1
        inputs = [np.sin(angle), np.cos(angle), 0.5]
        actions = [np.sin(angle * 2.0) * 0.8, np.cos(angle * 2.0) * 0.8]
        reward = 1.0 + np.sin(angle) * 0.5  # Dynamic periodic rewards
        buffer.add_frame(inputs, actions, reward)

    print(f"Replay buffer filled. Frame count: {buffer.count} / {buffer.capacity}")

    # 4. Perform GPU Evolutionary Step
    print(f"[INFO] Launching evolutionary mutation of {num_clones} clones on the GPU...")
    start_time = time.perf_counter()
    
    champion, fitness = solver.evolve(
        parent,
        buffer,
        num_clones=num_clones,
        mutation_rate=0.2,
        mutation_strength=0.15
    )
    
    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"[SUCCESS] GPU evolutionary evaluation finished in {elapsed:.3f} ms!")
    print(f"Champion Brain Fitness: {fitness:.4f}")

    # 5. Verify champion properties
    assert isinstance(champion, python_bridge.FastBrain), "Error: Champion is not a FastBrain instance!"
    assert champion.num_inputs == num_inputs, "Error: Inputs count mismatch!"
    assert champion.num_outputs == num_outputs, "Error: Outputs count mismatch!"
    assert champion.num_hidden == num_hidden, "Error: Hidden count mismatch!"
    
    print("\n[SUCCESS] GPU evolutionary forge verified. Ready for deep RL integration!")
    print("=" * 70)

if __name__ == "__main__":
    test_gpu_evolution()
