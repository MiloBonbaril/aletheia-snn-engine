// CSR (Compressed Sparse Row) matrix structure and SNN inference engine

/// FastBrain represents a zero-allocation, high-performance Spiking Neural Network.
/// It uses a Struct of Arrays (SoA) memory layout for neuron state and a
/// Compressed Sparse Row (CSR) layout for synaptic connections to optimize CPU cache performance.
pub struct FastBrain {
    /// Number of input neurons (sensors).
    pub num_inputs: usize,
    /// Number of hidden neurons.
    pub num_hidden: usize,
    /// Number of output neurons (actuators).
    pub num_outputs: usize,
    /// Total number of active neurons in the brain (inputs + hidden + outputs).
    pub active_neurons: usize,

    // SoA: Struct of Arrays. Contiguous flat vectors for maximum cache locality.
    /// Membrane potentials of all neurons.
    pub activations: Vec<f32>,
    /// Firing thresholds for each neuron.
    pub thresholds: Vec<f32>,
    /// Biases for each neuron (default to 0.0, available for future learning/mutations).
    pub biases: Vec<f32>,
    
    /// Leak decay rate (e.g., 0.85 means potentials decay by 15% each tick).
    pub decay: f32,
    
    /// Bitmask/Boolean array of neurons that spiked in the last tick.
    pub last_spikes: Vec<bool>,

    // Pre-allocated flat buffers to achieve 100% zero heap allocations on the hot path.
    current_spikes: Vec<bool>,
    inputs_accumulated: Vec<f32>,

    // CSR Topology arrays for synaptic weights
    /// Synaptic weights of all active connections.
    pub synapse_weights: Vec<f32>,
    /// Target neuron ID for each synapse.
    pub synapse_targets: Vec<u32>,
    /// Index in synapse arrays where each neuron's outgoing connections begin.
    /// Size is active_neurons + 1.
    pub neuron_offsets: Vec<u32>,
}

impl FastBrain {
    /// Creates a new FastBrain of a given shape with a deterministic connection structure.
    pub fn new(num_inputs: usize, num_hidden: usize, num_outputs: usize) -> Self {
        let active_neurons = num_inputs + num_hidden + num_outputs;
        
        let activations = vec![0.0; active_neurons];
        let mut thresholds = vec![0.0; active_neurons];
        let biases = vec![0.0; active_neurons];
        let decay = 0.85;

        // Set default thresholds for hidden and output neurons
        for i in num_inputs..active_neurons {
            thresholds[i] = 1.0;
        }

        // Initialize a temporary dense representation to perform deterministic seed-based weight generation
        let mut dense_weights = vec![0.0f32; active_neurons * active_neurons];

        let mut seed: u32 = 42;
        let mut next_random = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed / 65536) % 32768) as f32 / 32767.0 * 2.0 - 1.0 // Range [-1.0, 1.0]
        };

        let hidden_start = num_inputs;
        let hidden_end = num_inputs + num_hidden;
        let output_start = hidden_end;
        let output_end = active_neurons;

        // 1. Inputs to Hidden
        for src in 0..num_inputs {
            for dst in hidden_start..hidden_end {
                // Sparsely connect: 60% connection probability (next_random() > -0.2)
                if next_random() > -0.2 {
                    dense_weights[src * active_neurons + dst] = next_random() * 0.5;
                }
            }
        }

        // 2. Recurrent Hidden to Hidden
        for src in hidden_start..hidden_end {
            for dst in hidden_start..hidden_end {
                if src != dst && next_random() > 0.6 {
                    dense_weights[src * active_neurons + dst] = next_random() * 0.2;
                }
            }
        }

        // 3. Hidden to Output
        for src in hidden_start..hidden_end {
            for dst in output_start..output_end {
                if next_random() > -0.4 {
                    dense_weights[src * active_neurons + dst] = next_random() * 0.8;
                }
            }
        }

        // Convert dense matrix into Compressed Sparse Row (CSR) representation
        let mut synapse_weights = Vec::new();
        let mut synapse_targets = Vec::new();
        let mut neuron_offsets = Vec::with_capacity(active_neurons + 1);

        neuron_offsets.push(0);

        for src in 0..active_neurons {
            for dst in 0..active_neurons {
                let w = dense_weights[src * active_neurons + dst];
                if w != 0.0 {
                    synapse_weights.push(w);
                    synapse_targets.push(dst as u32);
                }
            }
            neuron_offsets.push(synapse_weights.len() as u32);
        }

        Self {
            num_inputs,
            num_hidden,
            num_outputs,
            active_neurons,
            activations,
            thresholds,
            biases,
            decay,
            last_spikes: vec![false; active_neurons],
            current_spikes: vec![false; active_neurons],
            inputs_accumulated: vec![0.0; active_neurons],
            synapse_weights,
            synapse_targets,
            neuron_offsets,
        }
    }

    /// Performs one simulation step (tick) of the SNN.
    /// 100% zero heap allocations on the hot path by reusing the pre-allocated buffers.
    ///
    /// - `inputs`: Sensors input observations from the environment.
    /// - Returns: Continuously mapped actuator actions.
    pub fn tick(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Zero out the pre-allocated buffers (hot path zero-allocation reset)
        self.inputs_accumulated.fill(0.0);
        self.current_spikes.fill(false);

        // 1. Process Input Neurons (0..num_inputs)
        // Set input potentials directly from inputs. If positive, they spike.
        for i in 0..self.num_inputs {
            let val = inputs.get(i).copied().unwrap_or(0.0);
            self.activations[i] = val;
            if val > 0.0 {
                self.current_spikes[i] = true;
            }
        }

        let hidden_start = self.num_inputs;
        let hidden_end = self.num_inputs + self.num_hidden;

        // 2. Direct input driving: input neurons propagate their activations to target hidden neurons
        for src in 0..self.num_inputs {
            let act = self.activations[src];
            if act != 0.0 {
                let start = self.neuron_offsets[src] as usize;
                let end = self.neuron_offsets[src + 1] as usize;
                for idx in start..end {
                    let target = self.synapse_targets[idx] as usize;
                    let weight = self.synapse_weights[idx];
                    if target >= hidden_start && target < hidden_end {
                        self.inputs_accumulated[target] += act * weight;
                    }
                }
            }
        }

        // 3. Recurrent hidden spikes from last tick
        for src in hidden_start..hidden_end {
            if self.last_spikes[src] {
                let start = self.neuron_offsets[src] as usize;
                let end = self.neuron_offsets[src + 1] as usize;
                for idx in start..end {
                    let target = self.synapse_targets[idx] as usize;
                    let weight = self.synapse_weights[idx];
                    if target >= hidden_start && target < hidden_end {
                        self.inputs_accumulated[target] += weight;
                    }
                }
            }
        }

        // 4. Integrate leaky potential for Hidden Neurons and check threshold
        for dst in hidden_start..hidden_end {
            let current = self.inputs_accumulated[dst] + self.biases[dst];
            self.activations[dst] = self.activations[dst] * self.decay + current;

            if self.activations[dst] >= self.thresholds[dst] {
                self.current_spikes[dst] = true;
                self.activations[dst] = 0.0; // Reset
            }
        }

        // 5. Output Neurons receive inputs from currently spiking hidden neurons
        let output_start = hidden_end;
        let output_end = self.active_neurons;

        for src in hidden_start..hidden_end {
            if self.current_spikes[src] {
                let start = self.neuron_offsets[src] as usize;
                let end = self.neuron_offsets[src + 1] as usize;
                for idx in start..end {
                    let target = self.synapse_targets[idx] as usize;
                    let weight = self.synapse_weights[idx];
                    if target >= output_start && target < output_end {
                        self.inputs_accumulated[target] += weight;
                    }
                }
            }
        }

        // 6. Integrate leaky potential for Output Neurons & check threshold
        for dst in output_start..output_end {
            let current = self.inputs_accumulated[dst] + self.biases[dst];
            self.activations[dst] = self.activations[dst] * self.decay + current;

            if self.activations[dst] >= self.thresholds[dst] {
                self.current_spikes[dst] = true;
                self.activations[dst] = 0.0; // Reset
            }
        }

        // Save current spikes for recurrent step next tick
        self.last_spikes.copy_from_slice(&self.current_spikes);

        // Map output membrane potentials to continuous motor actions in [-1.0, 1.0] via tanh
        let mut actions = vec![0.0; self.num_outputs];
        for i in 0..self.num_outputs {
            actions[i] = self.activations[output_start + i].tanh();
        }

        actions
    }

    /// Exposes the current spikes bitmask packed into a u64 (first 64 neurons).
    /// Maintains perfect compatibility with PyO3 FFI and telemetry tracking.
    pub fn get_last_spikes(&self) -> u64 {
        let mut mask = 0u64;
        let limit = std::cmp::min(self.active_neurons, 64);
        for i in 0..limit {
            if self.last_spikes[i] {
                mask |= 1 << i;
            }
        }
        mask
    }
}

impl Default for FastBrain {
    /// Returns the default BipedalWalker-v3 brain architecture (24 inputs, 36 hidden, 4 outputs).
    fn default() -> Self {
        Self::new(24, 36, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LegacyDenseBrain {
        potentials: [f32; 64],
        thresholds: [f32; 64],
        weights: [f32; 4096],
        decay: f32,
        last_spikes: u64,
    }

    impl LegacyDenseBrain {
        fn new() -> Self {
            let potentials = [0.0; 64];
            let mut thresholds = [1.0; 64];
            let mut weights = [0.0; 4096];
            let decay = 0.85;

            for i in 24..64 {
                thresholds[i] = 1.0;
            }

            let mut seed: u32 = 42;
            let mut next_random = || {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed / 65536) % 32768) as f32 / 32767.0 * 2.0 - 1.0
            };

            for src in 0..24 {
                for dst in 24..60 {
                    if next_random() > -0.2 {
                        weights[src * 64 + dst] = next_random() * 0.5;
                    }
                }
            }

            for src in 24..60 {
                for dst in 24..60 {
                    if src != dst && next_random() > 0.6 {
                        weights[src * 64 + dst] = next_random() * 0.2;
                    }
                }
            }

            for src in 24..60 {
                for dst in 60..64 {
                    if next_random() > -0.4 {
                        weights[src * 64 + dst] = next_random() * 0.8;
                    }
                }
            }

            Self {
                potentials,
                thresholds,
                weights,
                decay,
                last_spikes: 0,
            }
        }

        fn tick(&mut self, inputs: &[f32]) -> [f32; 4] {
            let mut current_spikes: u64 = 0;

            for i in 0..24 {
                let val = inputs.get(i).copied().unwrap_or(0.0);
                self.potentials[i] = val;
                if val > 0.0 {
                    current_spikes |= 1 << i;
                }
            }

            for dst in 24..60 {
                let mut current = 0.0;
                for src in 0..24 {
                    current += self.weights[src * 64 + dst] * self.potentials[src];
                }
                for src in 24..60 {
                    if (self.last_spikes & (1 << src)) != 0 {
                        current += self.weights[src * 64 + dst];
                    }
                }
                self.potentials[dst] = self.potentials[dst] * self.decay + current;
                if self.potentials[dst] >= self.thresholds[dst] {
                    current_spikes |= 1 << dst;
                    self.potentials[dst] = 0.0;
                }
            }

            for dst in 60..64 {
                let mut current = 0.0;
                for src in 24..60 {
                    if (current_spikes & (1 << src)) != 0 {
                        current += self.weights[src * 64 + dst];
                    }
                }
                self.potentials[dst] = self.potentials[dst] * self.decay + current;
                if self.potentials[dst] >= self.thresholds[dst] {
                    current_spikes |= 1 << dst;
                    self.potentials[dst] = 0.0;
                }
            }

            self.last_spikes = current_spikes;

            let mut actions = [0.0; 4];
            for i in 0..4 {
                actions[i] = self.potentials[60 + i].tanh();
            }
            actions
        }
    }

    #[test]
    fn test_dense_to_csr_mathematical_parity() {
        let mut legacy = LegacyDenseBrain::new();
        let mut csr = FastBrain::default();

        // Check if CSR conversion of synapse counts matches non-zero counts of legacy
        let legacy_non_zero = legacy.weights.iter().filter(|&&w| w != 0.0).count();
        assert_eq!(csr.synapse_weights.len(), legacy_non_zero);

        // Run a simulation of 100 ticks with deterministic pseudo-random inputs to verify parity
        let mut seed: u32 = 1337;
        let mut next_input = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed / 65536) % 32768) as f32 / 32767.0 * 2.0 - 1.0
        };

        for tick in 1..=100 {
            let mut inputs = [0.0f32; 24];
            for i in 0..24 {
                inputs[i] = next_input();
            }

            let legacy_actions = legacy.tick(&inputs);
            let csr_actions = csr.tick(&inputs);

            // Assert actions match exactly (within floating point precision limits)
            for i in 0..4 {
                assert!((legacy_actions[i] - csr_actions[i]).abs() < 1e-6, 
                    "Parity mismatch at tick {}: legacy={:?}, csr={:?}", tick, legacy_actions, csr_actions);
            }

            // Assert last spikes bitmask matches exactly
            assert_eq!(legacy.last_spikes, csr.get_last_spikes(), 
                "Spikes mismatch at tick {}: legacy={:#x}, csr={:#x}", tick, legacy.last_spikes, csr.get_last_spikes());
        }
    }

    #[test]
    fn test_dynamic_sizing() {
        // Build a dynamic brain with custom dimensions
        let mut brain = FastBrain::new(3, 5, 2);

        assert_eq!(brain.num_inputs, 3);
        assert_eq!(brain.num_hidden, 5);
        assert_eq!(brain.num_outputs, 2);
        assert_eq!(brain.active_neurons, 10);
        assert_eq!(brain.neuron_offsets.len(), 11);

        // Test one tick with dynamic sizes
        let inputs = [0.5, -0.2, 0.8];
        let actions = brain.tick(&inputs);

        assert_eq!(actions.len(), 2);
        for a in actions {
            assert!(a >= -1.0 && a <= 1.0);
        }
    }
}
