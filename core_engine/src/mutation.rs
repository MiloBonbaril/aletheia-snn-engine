use crate::brain::FastBrain;
use rand::Rng;
use rayon::prelude::*;

/// MutationEngine coordinates the evolutionary algorithm (weight mutations and parallel evaluations).
pub struct MutationEngine {
    /// Probability that a specific synapse weight undergoes mutation (e.g. 0.15).
    pub mutation_rate: f32,
    /// Maximum amplitude of mutation perturbation (e.g. 0.05).
    pub mutation_strength: f32,
}

impl MutationEngine {
    /// Creates a new MutationEngine.
    pub fn new(mutation_rate: f32, mutation_strength: f32) -> Self {
        Self {
            mutation_rate,
            mutation_strength,
        }
    }

    /// Perturbs the synaptic weights of a parent brain, returning a mutated clone.
    /// Operates entirely in contiguous memory layout, preserving maximum CPU cache locality.
    pub fn mutate(&self, parent: &FastBrain) -> FastBrain {
        let mut child = parent.clone();
        let mut rng = rand::thread_rng();

        for w in &mut child.synapse_weights {
            if rng.gen_range(0.0..1.0) < self.mutation_rate {
                // Apply a uniform random mutation weight adjustment
                *w += rng.gen_range(-self.mutation_strength..self.mutation_strength);
            }
        }

        // Reset the child's activations, spikes, and pre-allocated buffers
        child.activations.fill(0.0);
        child.last_spikes.fill(false);
        
        child
    }

    /// Evaluates multiple mutated clones in parallel using the CPU Rayon solver.
    /// Runs separate agnostically decoupled game environments concurrently on all CPU threads,
    /// returning the best mutated SNN clone along with its achieved fitness score.
    pub fn evolve<E>(
        &self,
        parent: &FastBrain,
        env_factory: &(dyn Fn() -> E + Sync),
        num_clones: usize,
        max_steps: usize,
    ) -> (FastBrain, f32)
    where
        E: crate::environment::Environment + Send,
    {
        // 1. Generate the generation pool of mutated clones in parallel
        let clones: Vec<FastBrain> = (0..num_clones)
            .into_par_iter()
            .map(|_| self.mutate(parent))
            .collect();

        // 2. Evaluate all clones in parallel
        let results: Vec<(FastBrain, f32)> = clones
            .into_par_iter()
            .map(|mut brain| {
                let mut env = env_factory();
                env.reset();

                let mut total_reward = 0.0;
                let mut state = env.get_state();

                for _ in 0..max_steps {
                    let actions = brain.tick(&state);
                    let (next_state, reward, done) = env.step(&actions);
                    total_reward += reward;
                    state = next_state;
                    if done {
                        break;
                    }
                }

                (brain, total_reward)
            })
            .collect();

        // 3. Selection: find the clone with the highest fitness score
        results
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| (parent.clone(), 0.0))
    }
}

// -----------------------------------------------------------------------------
// CUDA GPU Batch SNN propagation Solver (Active when the "cuda" feature is enabled)
// -----------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

    /// CudaSnnSolver orchestrates the offloading of SNN propagation batches to the GPU.
    pub struct CudaSnnSolver {
        device: Arc<CudaDevice>,
        function: CudaFunction,
    }

    impl CudaSnnSolver {
        /// Initializes the CUDA solver, loads the custom compiled PTX module, and gets the kernel function.
        pub fn new() -> Result<Self, cudarc::driver::driver_error::DriverError> {
            let device = CudaDevice::new(0)?;
            
            // Load the PTX compiled during the build phase
            let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/snn_kernel.ptx"));
            
            device.load_ptx(ptx_src.into(), "snn_module", &["compute_snn_kernel"])?;
            let function = device.get_func("snn_module", "compute_snn_kernel").unwrap();

            Ok(Self { device, function })
        }

        /// Simulates a single tick for a batch of clones in parallel on the GPU.
        /// Demonstrates GPU offloading of 1000+ SNN activations.
        pub fn step_batch(
            &self,
            num_neurons: usize,
            num_clones: usize,
            num_synapses: usize,
            membrane_potentials: &mut [f32],
            spike_out: &mut [i32],
            spike_in: &[i32],
            weights: &[f32],
            synapse_targets: &[u32],
            neuron_offsets: &[u32],
            thresholds: &[f32],
            beta: f32,
        ) -> Result<(), cudarc::driver::driver_error::DriverError> {
            // Allocate and transfer CPU arrays to GPU VRAM (device memory)
            let mut dev_potentials = self.device.htod_copy(membrane_potentials.to_vec())?;
            let mut dev_spike_out = self.device.alloc_zeros::<i32>(num_clones * num_neurons)?;
            let dev_spike_in = self.device.htod_copy(spike_in.to_vec())?;
            let dev_weights = self.device.htod_copy(weights.to_vec())?;
            
            // Cast u32 arrays to i32 for compatibility with standard CUDA integer types
            let targets_i32: Vec<i32> = synapse_targets.iter().map(|&x| x as i32).collect();
            let offsets_i32: Vec<i32> = neuron_offsets.iter().map(|&x| x as i32).collect();
            
            let dev_syn_targets = self.device.htod_copy(targets_i32)?;
            let dev_neur_offsets = self.device.htod_copy(offsets_i32)?;
            let dev_thresholds = self.device.htod_copy(thresholds.to_vec())?;

            let total_threads = (num_clones * num_neurons) as u32;
            
            // Configure threads and blocks optimally for high occupancy (blocks of 256 threads)
            let threads_per_block = 256;
            let blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };

            // Launch the batch CUDA kernel asynchronously
            unsafe {
                self.function.clone().launch(
                    config,
                    (
                        num_neurons as i32,
                        num_clones as i32,
                        num_synapses as i32,
                        &mut dev_potentials,
                        &mut dev_spike_out,
                        &dev_spike_in,
                        &dev_weights,
                        &dev_syn_targets,
                        &dev_neur_offsets,
                        &dev_thresholds,
                        beta,
                    ),
                )?;
            }

            // Retrieve outputs back from GPU VRAM to CPU memory
            let updated_potentials = self.device.sync_reclaim(dev_potentials)?;
            let computed_spikes = self.device.sync_reclaim(dev_spike_out)?;

            membrane_potentials.copy_from_slice(&updated_potentials);
            spike_out.copy_from_slice(&computed_spikes);

            Ok(())
        }
    }
}
