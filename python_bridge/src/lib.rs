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

    /// Get the number of inputs (sensor neurons).
    #[getter]
    pub fn num_inputs(&self) -> usize {
        self.active_brain.num_inputs
    }

    /// Get the number of hidden neurons.
    #[getter]
    pub fn num_hidden(&self) -> usize {
        self.active_brain.num_hidden
    }

    /// Get the number of outputs (actuator neurons).
    #[getter]
    pub fn num_outputs(&self) -> usize {
        self.active_brain.num_outputs
    }

    /// Saves the current SNN state to a pretty-printed JSON file on disk.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.active_brain.save_to_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save SNN state: {}", e)))
    }

    /// Loads an SNN state from a JSON file on disk into this existing brain instance,
    /// updating the active network parameterization atomically.
    pub fn load(&mut self, path: &str) -> PyResult<()> {
        let core = CoreFastBrain::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load SNN state: {}", e)))?;
        self.active_brain = core;
        Ok(())
    }

    /// Statically instantiates a brand new FastBrain and telemetry context directly from a saved JSON weights file on disk.
    #[staticmethod]
    pub fn load_from_file(path: &str) -> PyResult<Self> {
        let core = CoreFastBrain::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load SNN state from file: {}", e)))?;
        let telemetry = TelemetryHub::new();
        telemetry.start_background_loop();
        Ok(Self {
            active_brain: core,
            swapped_brain: Arc::new(ArcSwap::from_pointee(None)),
            telemetry,
        })
    }
}

/// Circular replay buffer exposed to Python.
#[pyclass]
#[derive(Clone)]
pub struct PhantomReplayBuffer {
    pub inner: core_engine::arena::PhantomReplayBuffer,
}

#[pymethods]
impl PhantomReplayBuffer {
    #[new]
    pub fn new(num_inputs: usize, num_outputs: usize, capacity: usize) -> Self {
        Self {
            inner: core_engine::arena::PhantomReplayBuffer::new(num_inputs, num_outputs, capacity),
        }
    }

    pub fn add_frame(&mut self, input: Vec<f32>, action: Vec<f32>, reward: f32) -> PyResult<()> {
        self.inner.add_frame(&input, &action, reward);
        Ok(())
    }

    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset();
        Ok(())
    }

    #[getter]
    pub fn num_inputs(&self) -> usize {
        self.inner.num_inputs
    }

    #[getter]
    pub fn num_outputs(&self) -> usize {
        self.inner.num_outputs
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    #[getter]
    pub fn write_idx(&self) -> usize {
        self.inner.write_idx
    }

    #[getter]
    pub fn count(&self) -> usize {
        self.inner.count
    }
}

/// High-speed GPU SNN evolutionary mutant evaluation solver.
#[cfg(feature = "cuda")]
#[pyclass]
pub struct CudaSnnSolver {
    pub solver: core_engine::mutation::cuda::CudaSnnSolver,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl CudaSnnSolver {
    #[new]
    pub fn new() -> PyResult<Self> {
        let solver = core_engine::mutation::cuda::CudaSnnSolver::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA driver error: {:?}", e)))?;
        Ok(Self { solver })
    }

    /// Evaluates custom mutated weights flat array on the circular replay buffer.
    pub fn evaluate_phantom_mutants(
        &self,
        parent: &FastBrain,
        replay_buffer: &PhantomReplayBuffer,
        mutated_weights: Vec<f32>,
        num_clones: usize,
    ) -> PyResult<(usize, f32)> {
        let (champion_id, max_fitness) = self.solver.evaluate_phantom_mutants(
            &parent.active_brain,
            &replay_buffer.inner,
            &mutated_weights,
            num_clones,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA evaluation failed: {:?}", e)))?;
        Ok((champion_id, max_fitness))
    }

    /// Generates mutant clones and evaluates them on the GPU, returning the champion FastBrain.
    pub fn evolve(
        &self,
        parent: &FastBrain,
        replay_buffer: &PhantomReplayBuffer,
        num_clones: usize,
        mutation_rate: f32,
        mutation_strength: f32,
    ) -> PyResult<(FastBrain, f32)> {
        if num_clones == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("num_clones must be > 0"));
        }

        let num_synapses = parent.active_brain.synapse_weights.len();
        let mut mutated_weights = vec![0.0; num_clones * num_synapses];
        
        // Generate weight mutations using a fast CPU-parallel block
        use rayon::prelude::*;
        
        mutated_weights.par_chunks_mut(num_synapses).for_each(|chunk| {
            let mut rng = rand::thread_rng();
            for i in 0..num_synapses {
                let mut w = parent.active_brain.synapse_weights[i];
                if rand::Rng::gen_range(&mut rng, 0.0..1.0) < mutation_rate {
                    w += rand::Rng::gen_range(&mut rng, -mutation_strength..mutation_strength);
                }
                chunk[i] = w;
            }
        });

        // Evaluate mutants on the GPU circular buffer
        let (champion_id, max_fitness) = self.solver.evaluate_phantom_mutants(
            &parent.active_brain,
            &replay_buffer.inner,
            &mutated_weights,
            num_clones,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA evaluation failed: {:?}", e)))?;

        // Construct the new champion FastBrain with the champion's weights
        let mut champion_brain = parent.active_brain.clone();
        let champ_offset = champion_id * num_synapses;
        champion_brain.synapse_weights.copy_from_slice(&mutated_weights[champ_offset..(champ_offset + num_synapses)]);
        
        // Reset states of the new champion brain
        champion_brain.activations.fill(0.0);
        champion_brain.last_spikes.fill(false);

        let telemetry = TelemetryHub::new();
        telemetry.start_background_loop();

        let py_champion = FastBrain {
            active_brain: champion_brain,
            swapped_brain: Arc::new(ArcSwap::from_pointee(None)),
            telemetry,
        };

        Ok((py_champion, max_fitness))
    }
}

/// The Aletheia SNN Python module entry point.
#[pymodule]
fn python_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastBrain>()?;
    m.add_class::<PhantomReplayBuffer>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<CudaSnnSolver>()?;
    Ok(())
}
