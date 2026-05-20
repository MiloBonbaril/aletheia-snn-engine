use pyo3::prelude::*;
use core_engine::brain::FastBrain as CoreFastBrain;
use core_engine::telemetry::TelemetryHub;
use arc_swap::ArcSwap;
use std::sync::Arc;

/// PyO3 wrapped FastBrain class for Gymnasium interaction.
#[pyclass]
pub struct FastBrain {
    active_brain: CoreFastBrain,
    swapped_brain: Arc<ArcSwap<Option<CoreFastBrain>>>,
    telemetry: TelemetryHub,
}

#[pymethods]
impl FastBrain {
    /// Create a new FastBrain instance with optional custom dimensions.
    #[new]
    #[pyo3(signature = (num_inputs = 24, num_hidden = 36, num_outputs = 4))]
    pub fn new(num_inputs: usize, num_hidden: usize, num_outputs: usize) -> Self {
        let telemetry = TelemetryHub::new();
        // Start the background telemetry loop (Simulated telemetry server thread)
        telemetry.start_background_loop();
        
        Self {
            active_brain: CoreFastBrain::new(num_inputs, num_hidden, num_outputs),
            swapped_brain: Arc::new(ArcSwap::from_pointee(None)),
            telemetry,
        }
    }

    /// Ingests sensor input observations from Python, triggers SNN propagation
    /// to compute the continuous motor actions, pushes the spikes bitmask
    /// to telemetry, and returns the actions.
    pub fn tick(&mut self, inputs: Vec<f32>) -> PyResult<Vec<f32>> {
        // 0. Double-buffered Lock-Free Swapping:
        // Check if the background mutation thread has staged a new mutated brain in our slot.
        if let Some(ref new_core) = **self.swapped_brain.load() {
            // Hot-swap the active brain structure atomically off the hot path
            self.active_brain = new_core.clone();
            // Clear the slot so we don't copy it again next tick
            self.swapped_brain.store(Arc::new(None));
        }

        if inputs.len() != self.active_brain.num_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("FastBrain requires exactly {} inputs, got {}", self.active_brain.num_inputs, inputs.len())
            ));
        }

        // 1. Python says: "Here is the state of the environment (floats)"
        // 2. PyO3 passes it directly to core_engine in RAM
        // 3. FastBrain calculates spike propagation and computes actions
        let actions = self.active_brain.tick(&inputs);

        // 4. FastBrain pushes the active neurons bitmask to the telemetry hub
        let spikes_bitmask = self.active_brain.get_last_spikes();
        self.telemetry.record_spikes(spikes_bitmask);

        // 5. Python retrieves the 4 actions to drive the robot
        Ok(actions.to_vec())
    }

    /// Retrieve the current active spikes bitmask.
    pub fn get_last_spikes(&self) -> u64 {
        self.active_brain.get_last_spikes()
    }

    /// Retrieve the spikes bitmask recorded by the telemetry hub.
    pub fn get_telemetry_spikes(&self) -> u64 {
        self.telemetry.get_last_spikes()
    }

    /// Swaps the current brain's weights and structure atomically with another FastBrain's active brain.
    /// This is used by the background evolutionary training thread to replace the active brain lock-free!
    pub fn swap_brain(&self, new_brain: &FastBrain) -> PyResult<()> {
        let new_core = new_brain.active_brain.clone();
        self.swapped_brain.store(Arc::new(Some(new_core)));
        Ok(())
    }

    /// Generates a mutated copy of the current brain structure with the given rate and strength.
    /// Allows seamless, high-performance evolutionary mutation operations directly from Python!
    pub fn mutate(&self, mutation_rate: f32, mutation_strength: f32) -> Self {
        let mutation_engine = core_engine::mutation::MutationEngine::new(mutation_rate, mutation_strength);
        let mutated_core = mutation_engine.mutate(&self.active_brain);
        
        let telemetry = TelemetryHub::new();
        telemetry.start_background_loop();
        
        Self {
            active_brain: mutated_core,
            swapped_brain: Arc::new(ArcSwap::from_pointee(None)),
            telemetry,
        }
    }
}

/// The Aletheia SNN Python module entry point.
#[pymodule]
fn python_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastBrain>()?;
    Ok(())
}
