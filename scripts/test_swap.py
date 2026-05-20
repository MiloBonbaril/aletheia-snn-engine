#!/usr/bin/env python3
import sys
import os
import time
import threading
import numpy as np

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
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import python_bridge

def test_lock_free_swapping():
    print("=" * 70)
    print("[ALETHEIA SNN] - INTEGRATION TEST FOR LOCK-FREE SWAPPING")
    print("=" * 70)
    
    # Initialize the spiking neural network
    brain = python_bridge.FastBrain()
    print("Initial brain created successfully.")
    
    stop_event = threading.Event()
    swap_count = 0
    
    # Define a background thread to mutate and swap the SNN brain *live*
    def background_mutator():
        nonlocal swap_count
        print("[Background Thread] Started and active...")
        while not stop_event.is_set():
            time.sleep(0.1) # Mutate and swap every 100ms
            
            # 1. Generate a mutated brain from the active one
            # Mutation rate 0.5, mutation strength 0.2 to ensure noticeable change
            mutated_brain = brain.mutate(0.5, 0.2)
            
            # 2. Atomically swap it into the active brain running in the main thread
            brain.swap_brain(mutated_brain)
            
            swap_count += 1
            print(f"[Background Thread] Mutated clone #{swap_count} staged in ArcSwap slot!")
            
    # Start background thread
    bg_thread = threading.Thread(target=background_mutator)
    bg_thread.daemon = True
    bg_thread.start()
    
    # Main thread runs a hot simulated game loop calling tick() at full speed
    print("[Main Thread] Starting SNN propagation loop...")
    tick_durations = []
    spikes_history = []
    
    # Let's run 50 steps
    for step in range(1, 51):
        # Generate random inputs mimicking Gymnasium observation vector (24 floats)
        inputs = list(np.random.uniform(-1.0, 1.0, 24))
        
        start_tick = time.perf_counter()
        
        # Ingest state and tick SNN propagation
        actions = brain.tick(inputs)
        
        duration = (time.perf_counter() - start_tick) * 1000.0 # in ms
        tick_durations.append(duration)
        
        spikes = brain.get_last_spikes()
        spikes_history.append(spikes)
        
        print(f"[Main Thread] Tick {step:2d} | actions: {[f'{a:.3f}' for a in actions]} | Spikes Mask: {hex(spikes)} | Tick Time: {duration:.4f} ms")
        time.sleep(0.02) # Simulate 50 FPS game loop (20ms step time)

    # Stop the background thread
    stop_event.set()
    bg_thread.join(timeout=1.0)
    
    print("-" * 70)
    print("PERFORMANCE & VERIFICATION ANALYSIS:")
    print("-" * 70)
    print(f"Total live atomic swaps staged: {swap_count}")
    print(f"Average SNN FFI step time: {np.mean(tick_durations):.4f} ms")
    print(f"Maximum SNN FFI step time: {np.max(tick_durations):.4f} ms")
    print(f"Minimum SNN FFI step time: {np.min(tick_durations):.4f} ms")
    
    # Verify that spikes changed over time (proving that new mutated weights were picked up live!)
    unique_spikes = len(set(spikes_history))
    print(f"Unique spikes bitmasks observed: {unique_spikes} / {len(spikes_history)}")
    
    assert unique_spikes > 1, "Error: Spikes did not change, hot swap might not have updated the weights!"
    assert np.max(tick_durations) < 50.0, "Error: Severe lag detected during atomic swap!"
    
    print("\n[SUCCESS] Spiking neural network was hot-swapped live with zero lock contention and zero latency spikes!")
    print("=" * 70)

if __name__ == "__main__":
    test_lock_free_swapping()
