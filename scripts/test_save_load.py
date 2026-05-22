#!/usr/bin/env python3
import os
import sys
import numpy as np

# Ensure we can import the compiled python_bridge
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import python_bridge
except ImportError:
    print("[ERROR] python_bridge is not compiled or not in path.")
    sys.exit(1)

def run_parity_test():
    print("=" * 60)
    print("[TEST] STARTING SNN PERSISTENCE PARITY TEST")
    print("=" * 60)

    # 1. Initialize SNN
    inputs_dim = 24
    hidden_dim = 36
    outputs_dim = 4
    brain_a = python_bridge.FastBrain(inputs_dim, hidden_dim, outputs_dim)
    
    print(f"SNN dimensions successfully queried via PyO3 getters:")
    print(f"  - num_inputs: {brain_a.num_inputs}")
    print(f"  - num_hidden: {brain_a.num_hidden}")
    print(f"  - num_outputs: {brain_a.num_outputs}")
    
    assert brain_a.num_inputs == inputs_dim
    assert brain_a.num_hidden == hidden_dim
    assert brain_a.num_outputs == outputs_dim

    # 2. Run simulation ticks to get baseline actions and active potentials
    print("\nRunning baseline simulation ticks...")
    np.random.seed(42)
    inputs_seq = [np.random.uniform(-1.0, 1.0, inputs_dim).tolist() for _ in range(20)]
    
    baseline_actions = []
    for inp in inputs_seq:
        actions = brain_a.tick(inp)
        baseline_actions.append(actions)

    # 3. Save SNN state to disk
    os.makedirs("weights", exist_ok=True)
    temp_json = "weights/test_parity.json"
    print(f"\nSaving SNN brain_a state to disk at '{temp_json}'...")
    brain_a.save(temp_json)

    # Verify JSON file exists and contains data
    assert os.path.exists(temp_json)
    print(f"JSON File Size: {os.path.getsize(temp_json)} bytes")

    # 4. Load SNN state into a brand new instance using the static factory method
    print(f"\nLoading pre-trained SNN state into brain_b from '{temp_json}'...")
    brain_b = python_bridge.FastBrain.load_from_file(temp_json)
    
    assert brain_b.num_inputs == inputs_dim
    assert brain_b.num_hidden == hidden_dim
    assert brain_b.num_outputs == outputs_dim

    # 5. Run the exact same sequence of inputs through brain_b and check mathematical parity
    print("\nRunning parity simulation on loaded brain_b...")
    loaded_actions = []
    for inp in inputs_seq:
        actions = brain_b.tick(inp)
        loaded_actions.append(actions)

    # Validate mathematical parity
    print("\nComparing outputs:")
    mismatch_count = 0
    for tick, (act_a, act_b) in enumerate(zip(baseline_actions, loaded_actions)):
        diff = np.abs(np.array(act_a) - np.array(act_b))
        max_diff = np.max(diff)
        if max_diff > 1e-7:
            print(f"  [FAIL] Tick {tick:02d} mismatch: Max Diff = {max_diff:.2e}")
            mismatch_count += 1
        else:
            print(f"  [PASS] Tick {tick:02d} perfect match (Max Diff: {max_diff:.2e})")

    assert mismatch_count == 0, "Test failed: Saved/Loaded actions did not match baseline actions!"
    print("\n[SUCCESS] Statically loaded brain_b matches brain_a perfectly to the decimal point!")

    # 6. Test instance-level .load() method
    print("\nTesting instance-level .load() method...")
    # Create a mutated copy of brain_a
    brain_mutated = brain_a.mutate(0.5, 0.5)
    mutated_actions = [brain_mutated.tick(inp) for inp in inputs_seq]
    
    # Assert that the mutated brain produces different actions
    mutated_diff = np.max(np.abs(np.array(baseline_actions) - np.array(mutated_actions)))
    print(f"  Mutated brain difference vs baseline: {mutated_diff:.4f}")
    assert mutated_diff > 1e-4, "Mutated brain must behave differently!"

    # Load original weights back into the mutated instance
    print(f"  Atomically loading original weights into brain_mutated using .load()...")
    brain_mutated.load(temp_json)
    
    # Run the sequence again on the reloaded brain
    reloaded_actions = []
    for inp in inputs_seq:
        actions = brain_mutated.tick(inp)
        reloaded_actions.append(actions)
        
    reloaded_diff = np.max(np.abs(np.array(baseline_actions) - np.array(reloaded_actions)))
    print(f"  Reloaded brain difference vs baseline: {reloaded_diff:.2e}")
    assert reloaded_diff < 1e-7, "Reloaded instance does not match baseline!"
    print("[SUCCESS] Instance-level .load() restored perfect mathematical parity!")

    # Cleanup
    if os.path.exists(temp_json):
        os.remove(temp_json)
        print("\nCleaned up temporary JSON weights file.")

    print("=" * 60)
    print("[TEST] ALL PERSISTENCE PARITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_parity_test()
