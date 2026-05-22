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
        pub device: Arc<CudaDevice>,
        pub function: CudaFunction,
        pub eval_function: CudaFunction,
        pub reduce_function: CudaFunction,
    }

    impl CudaSnnSolver {
        /// Initializes the CUDA solver, loads the custom compiled PTX module, and gets the kernel functions.
        pub fn new() -> Result<Self, cudarc::driver::DriverError> {
            let device = CudaDevice::new(0)?;
            
            // Load the PTX compiled during the build phase
            let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/snn_kernel.ptx"));
            
            device.load_ptx(
                ptx_src.into(),
                "snn_module",
                &[
                    "compute_snn_kernel",
                    "evaluate_phantom_mutants_kernel",
                    "find_champion_kernel",
                ],
            )?;
            let function = device.get_func("snn_module", "compute_snn_kernel").unwrap();
            let eval_function = device.get_func("snn_module", "evaluate_phantom_mutants_kernel").unwrap();
            let reduce_function = device.get_func("snn_module", "find_champion_kernel").unwrap();

            Ok(Self {
                device,
                function,
                eval_function,
                reduce_function,
            })
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
        ) -> Result<(), cudarc::driver::DriverError> {
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

        /// Simulates a batch of mutant SNN clones sequentially on the circular Replay Buffer
        /// and finds the best mutant clone ID along with its fitness score.
        pub fn evaluate_phantom_mutants(
            &self,
            parent: &FastBrain,
            replay_buffer: &crate::arena::PhantomReplayBuffer,
            mutated_weights: &[f32], // Flat vector of size num_clones * num_synapses
            num_clones: usize,
        ) -> Result<(usize, f32), cudarc::driver::DriverError> {
            if num_clones == 0 {
                return Ok((0, 0.0));
            }

            let num_neurons = parent.active_neurons;
            let num_synapses = parent.synapse_weights.len();

            // Circular buffer variables
            let capacity = replay_buffer.capacity;
            let count = replay_buffer.count;
            let write_idx = replay_buffer.write_idx;
            let oldest_idx = if count < capacity { 0 } else { write_idx };
            let history_len = count;

            if history_len == 0 {
                return Ok((0, 0.0));
            }

            // 1. Pack and transfer variables to GPU VRAM
            // Dims array: size 10
            let decay_bits = parent.decay.to_bits() as i32;
            let dims = [
                parent.num_inputs as i32,
                parent.num_hidden as i32,
                parent.num_outputs as i32,
                num_neurons as i32,
                num_clones as i32,
                num_synapses as i32,
                history_len as i32,
                oldest_idx as i32,
                capacity as i32,
                decay_bits,
            ];
            let dev_dims = self.device.htod_copy(dims.to_vec())?;

            // Transfer inputs, actions, rewards
            let dev_inputs = self.device.htod_copy(replay_buffer.inputs.clone())?;
            let dev_actions = self.device.htod_copy(replay_buffer.actions.clone())?;
            let dev_rewards = self.device.htod_copy(replay_buffer.rewards.clone())?;
            let dev_weights = self.device.htod_copy(mutated_weights.to_vec())?;

            // Topology (offsets + targets) packed in a single contiguous i32 array
            let targets_i32: Vec<i32> = parent.synapse_targets.iter().map(|&x| x as i32).collect();
            let offsets_i32: Vec<i32> = parent.neuron_offsets.iter().map(|&x| x as i32).collect();
            let mut topology_data = Vec::with_capacity(offsets_i32.len() + targets_i32.len());
            topology_data.extend_from_slice(&offsets_i32);
            topology_data.extend_from_slice(&targets_i32);
            let dev_topology = self.device.htod_copy(topology_data)?;

            // Neuron params (thresholds + biases) packed in a single contiguous f32 array
            let mut neuron_params = Vec::with_capacity(parent.thresholds.len() + parent.biases.len());
            neuron_params.extend_from_slice(&parent.thresholds);
            neuron_params.extend_from_slice(&parent.biases);
            let dev_neuron_params = self.device.htod_copy(neuron_params)?;

            // 2. Allocate workspace memory on GPU VRAM
            // Single contiguous workspace array of size 4 * num_clones * num_neurons
            let mut dev_fitness = self.device.alloc_zeros::<f32>(num_clones)?;
            let mut dev_workspace = self.device.alloc_zeros::<f32>(4 * num_clones * num_neurons)?;

            // 3. Launch Simulation Kernel
            let threads_per_block = 256;
            let blocks = (num_clones + threads_per_block - 1) / threads_per_block;
            let eval_config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.eval_function.clone().launch(
                    eval_config,
                    (
                        &dev_dims,
                        &dev_inputs,
                        &dev_actions,
                        &dev_rewards,
                        &dev_weights,
                        &dev_topology,
                        &dev_neuron_params,
                        &mut dev_fitness,
                        &mut dev_workspace,
                    ),
                )?;
            }

            // 4. Find the champion: Use GPU reduction if num_clones <= 1024, else fallback to CPU max
            let champion_id;
            let max_fitness;

            if num_clones <= 1024 {
                let mut dev_max_fitness = self.device.alloc_zeros::<f32>(1)?;
                let mut dev_champion_id = self.device.alloc_zeros::<i32>(1)?;

                let reduce_config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    self.reduce_function.clone().launch(
                        reduce_config,
                        (
                            &dev_fitness,
                            num_clones as i32,
                            &mut dev_max_fitness,
                            &mut dev_champion_id,
                        ),
                    )?;
                }

                let max_fitness_vec = self.device.sync_reclaim(dev_max_fitness)?;
                let champion_id_vec = self.device.sync_reclaim(dev_champion_id)?;
                champion_id = champion_id_vec[0] as usize;
                max_fitness = max_fitness_vec[0];
            } else {
                let fitness_vec = self.device.sync_reclaim(dev_fitness)?;
                let (best_idx, &best_fit) = fitness_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));
                champion_id = best_idx;
                max_fitness = best_fit;
            }

            Ok((champion_id, max_fitness))
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::cuda::CudaSnnSolver;
    use crate::brain::FastBrain;
    use crate::arena::PhantomReplayBuffer;
    use cudarc::driver::LaunchAsync;

    #[test]
    fn test_gpu_cpu_mathematical_parity() {
        let num_inputs = 3;
        let num_hidden = 4;
        let num_outputs = 2;
        let capacity = 5;
        let num_clones = 3;

        let mut parent = FastBrain::new(num_inputs, num_hidden, num_outputs);
        // Ensure STDP is not interfering or let it adapt
        parent.stdp_a_plus = 0.0;
        parent.stdp_a_minus = 0.0;

        let mut buffer = PhantomReplayBuffer::new(num_inputs, num_outputs, capacity);
        // Fill buffer with deterministic dummy data
        for i in 0..capacity {
            let input = vec![0.1 * i as f32, -0.2 * i as f32, 0.5];
            let action = vec![(i as f32 * 0.1).tanh(), (i as f32 * -0.15).tanh()];
            let reward = 1.5 + (i as f32 * 0.2);
            buffer.add_frame(&input, &action, reward);
        }

        // Generate mutated weights
        let num_synapses = parent.synapse_weights.len();
        let mut mutated_weights = vec![0.0; num_clones * num_synapses];
        for c in 0..num_clones {
            let offset = c * num_synapses;
            for i in 0..num_synapses {
                let w = parent.synapse_weights[i] + (c as f32 * 0.1) - (i as f32 * 0.05);
                mutated_weights[offset + i] = w;
            }
        }

        // 1. Evaluate on GPU
        let solver = CudaSnnSolver::new().expect("Failed to initialize CudaSnnSolver");
        
        let num_neurons = parent.active_neurons;
        let oldest_idx = if buffer.count < capacity { 0 } else { buffer.write_idx };
        let history_len = buffer.count;

        let dev_inputs = solver.device.htod_copy(buffer.inputs.clone()).unwrap();
        let dev_actions = solver.device.htod_copy(buffer.actions.clone()).unwrap();
        let dev_rewards = solver.device.htod_copy(buffer.rewards.clone()).unwrap();
        let dev_weights = solver.device.htod_copy(mutated_weights.clone()).unwrap();

        let targets_i32: Vec<i32> = parent.synapse_targets.iter().map(|&x| x as i32).collect();
        let offsets_i32: Vec<i32> = parent.neuron_offsets.iter().map(|&x| x as i32).collect();
        let mut topology_data = Vec::new();
        topology_data.extend_from_slice(&offsets_i32);
        topology_data.extend_from_slice(&targets_i32);
        let dev_topology = solver.device.htod_copy(topology_data).unwrap();

        let mut neuron_params = Vec::new();
        neuron_params.extend_from_slice(&parent.thresholds);
        neuron_params.extend_from_slice(&parent.biases);
        let dev_neuron_params = solver.device.htod_copy(neuron_params).unwrap();

        let mut dev_fitness = solver.device.alloc_zeros::<f32>(num_clones).unwrap();
        let mut dev_workspace = solver.device.alloc_zeros::<f32>(4 * num_clones * num_neurons).unwrap();

        let decay_bits = parent.decay.to_bits() as i32;
        let dims = [
            num_inputs as i32,
            num_hidden as i32,
            num_outputs as i32,
            num_neurons as i32,
            num_clones as i32,
            num_synapses as i32,
            history_len as i32,
            oldest_idx as i32,
            capacity as i32,
            decay_bits,
        ];
        let dev_dims = solver.device.htod_copy(dims.to_vec()).unwrap();

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (num_clones as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            solver.eval_function.clone().launch(
                config,
                (
                    &dev_dims,
                    &dev_inputs,
                    &dev_actions,
                    &dev_rewards,
                    &dev_weights,
                    &dev_topology,
                    &dev_neuron_params,
                    &mut dev_fitness,
                    &mut dev_workspace,
                ),
            ).unwrap();
        }

        let gpu_fitness_results = solver.device.sync_reclaim(dev_fitness).unwrap();

        // 2. Evaluate on CPU (sequential reference)
        let mut cpu_fitnesses = vec![0.0; num_clones];
        for c in 0..num_clones {
            let mut clone = parent.clone();
            let offset = c * num_synapses;
            clone.synapse_weights.copy_from_slice(&mutated_weights[offset..(offset + num_synapses)]);
            clone.activations.fill(0.0);
            clone.last_spikes.fill(false);

            let mut fitness = 0.0;
            for t in 0..history_len {
                let frame_idx = (oldest_idx + t) % capacity;
                let input_start = frame_idx * num_inputs;
                let input = &buffer.inputs[input_start..(input_start + num_inputs)];
                let action_hist = &buffer.actions[(frame_idx * num_outputs)..(frame_idx * num_outputs + num_outputs)];
                let reward = buffer.rewards[frame_idx];

                let actions = clone.tick(input);

                let mut frame_dot = 0.0;
                for o in 0..num_outputs {
                    frame_dot += actions[o] * action_hist[o];
                }
                fitness += reward * frame_dot;
            }
            cpu_fitnesses[c] = fitness;
        }

        // Compare GPU vs CPU fitnesses
        for c in 0..num_clones {
            let diff = (gpu_fitness_results[c] - cpu_fitnesses[c]).abs();
            println!("Clone {}: GPU = {}, CPU = {}, Diff = {}", c, gpu_fitness_results[c], cpu_fitnesses[c], diff);
            assert!(diff < 1e-4, "Clone {} fitness does not match! GPU: {}, CPU: {}", c, gpu_fitness_results[c], cpu_fitnesses[c]);
        }
    }
}
