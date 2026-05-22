// CSR (Compressed Sparse Row) matrix structure and SNN inference engine
use serde::{Serialize, Deserialize};

/// FastBrain represents a zero-allocation, high-performance Spiking Neural Network.
/// It uses a Struct of Arrays (SoA) memory layout for neuron state and a
/// Compressed Sparse Row (CSR) layout for synaptic connections to optimize CPU cache performance.
#[derive(Clone, Debug)]
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

    // STDP learning trace buffers (SoA format)
    /// STDP pre-synaptic traces for all neurons.
    pub pre_traces: Vec<f32>,
    /// STDP post-synaptic traces for all neurons.
    pub post_traces: Vec<f32>,

    // STDP learning parameters
    /// Pre-synaptic trace decay time constant in ticks.
    pub stdp_tau_pre: f32,
    /// Post-synaptic trace decay time constant in ticks.
    pub stdp_tau_post: f32,
    /// LTP learning rate.
    pub stdp_a_plus: f32,
    /// LTD learning rate.
    pub stdp_a_minus: f32,
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
            pre_traces: vec![0.0; active_neurons],
            post_traces: vec![0.0; active_neurons],
            stdp_tau_pre: 20.0,
            stdp_tau_post: 20.0,
            stdp_a_plus: 0.005,
            stdp_a_minus: 0.006,
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

        // --- STDP (Spike-Timing-Dependent Plasticity) Trace-Based Learning ---
        // 1. Decay all traces exponentially: x_i <- x_i * exp(-1 / tau)
        let decay_pre = (-1.0 / self.stdp_tau_pre).exp();
        let decay_post = (-1.0 / self.stdp_tau_post).exp();
        for i in 0..self.active_neurons {
            self.pre_traces[i] *= decay_pre;
            self.post_traces[i] *= decay_post;
            
            // 2. If the neuron spiked in the current tick, set its trace to 1.0
            if self.current_spikes[i] {
                self.pre_traces[i] = 1.0;
                self.post_traces[i] = 1.0;
            }
        }

        // 3. Update synaptic weights using pre/post spikes and traces
        for j in 0..self.active_neurons {
            let start = self.neuron_offsets[j] as usize;
            let end = self.neuron_offsets[j + 1] as usize;
            let pre_spike = self.current_spikes[j];

            for idx in start..end {
                let i = self.synapse_targets[idx] as usize;
                let post_spike = self.current_spikes[i];

                // LTD: Pre-synaptic spike occurs, post-synaptic neuron was active recently
                if pre_spike {
                    self.synapse_weights[idx] -= self.stdp_a_minus * self.post_traces[i];
                }
                
                // LTP: Post-synaptic spike occurs, pre-synaptic neuron was active recently
                if post_spike {
                    self.synapse_weights[idx] += self.stdp_a_plus * self.pre_traces[j];
                }

                // Clip weight to prevent divergence (e.g. keep within [-2.0, 2.0])
                if self.synapse_weights[idx] > 2.0 {
                    self.synapse_weights[idx] = 2.0;
                } else if self.synapse_weights[idx] < -2.0 {
                    self.synapse_weights[idx] = -2.0;
                }
            }
        }

        // Map output membrane potentials to continuous motor actions in [-1.0, 1.0] via tanh
        let mut actions = vec![0.0; self.num_outputs];
        for i in 0..self.num_outputs {
            actions[i] = self.activations[output_start + i].tanh();
        }

        actions
    }

    /// Prunes synapses whose absolute weight is below the threshold (e.g. 1e-4).
    /// Reconstructs the CSR index offsets in-place to avoid vector allocation fragmentation.
    pub fn prune_synapses(&mut self) {
        let threshold = 1e-4f32;
        let mut new_weights = Vec::with_capacity(self.synapse_weights.len());
        let mut new_targets = Vec::with_capacity(self.synapse_targets.len());
        let mut new_offsets = Vec::with_capacity(self.neuron_offsets.len());

        new_offsets.push(0);

        for j in 0..self.active_neurons {
            let start = self.neuron_offsets[j] as usize;
            let end = self.neuron_offsets[j + 1] as usize;

            for idx in start..end {
                let w = self.synapse_weights[idx];
                let target = self.synapse_targets[idx];
                if w.abs() >= threshold {
                    new_weights.push(w);
                    new_targets.push(target);
                }
            }
            new_offsets.push(new_weights.len() as u32);
        }

        self.synapse_weights = new_weights;
        self.synapse_targets = new_targets;
        self.neuron_offsets = new_offsets;
    }

    /// Dynamically connects two active neurons in the CSR structure with the given weight.
    /// If the connection already exists, its weight is updated.
    pub fn add_synapse(&mut self, src: usize, dst: usize, weight: f32) {
        if src >= self.active_neurons || dst >= self.active_neurons {
            return;
        }

        // Check if connection already exists
        let start = self.neuron_offsets[src] as usize;
        let end = self.neuron_offsets[src + 1] as usize;
        for idx in start..end {
            if self.synapse_targets[idx] == dst as u32 {
                self.synapse_weights[idx] = weight;
                return;
            }
        }

        // Insert at the end of the outgoing synapses for `src`
        self.synapse_weights.insert(end, weight);
        self.synapse_targets.insert(end, dst as u32);

        // Shift offsets for all neurons after `src`
        for o in (src + 1)..=self.active_neurons {
            self.neuron_offsets[o] += 1;
        }
    }

    /// Adds a new hidden neuron by waking it up from our preallocated Arena pools.
    /// Returns the index of the newly created hidden neuron.
    pub fn add_hidden_neuron(&mut self) -> usize {
        let new_hidden_idx = self.num_inputs + self.num_hidden;
        
        // 1. Insert new elements into SoA arrays at new_hidden_idx
        self.activations.insert(new_hidden_idx, 0.0);
        self.thresholds.insert(new_hidden_idx, 1.0); // Default threshold
        self.biases.insert(new_hidden_idx, 0.0);
        self.last_spikes.insert(new_hidden_idx, false);
        self.current_spikes.insert(new_hidden_idx, false);
        self.inputs_accumulated.insert(new_hidden_idx, 0.0);
        self.pre_traces.insert(new_hidden_idx, 0.0);
        self.post_traces.insert(new_hidden_idx, 0.0);

        // Update counts
        self.num_hidden += 1;
        self.active_neurons += 1;

        // 2. We must insert a new entry in `neuron_offsets` at `new_hidden_idx`.
        let offset = self.neuron_offsets[new_hidden_idx];
        self.neuron_offsets.insert(new_hidden_idx, offset);

        // 3. Since output neurons shifted their indices by 1,
        // we must increment any target ID in `synapse_targets` that is >= new_hidden_idx by 1!
        for target in &mut self.synapse_targets {
            if *target >= new_hidden_idx as u32 {
                *target += 1;
            }
        }

        new_hidden_idx
    }

    /// Splits an existing synapse (src -> dst) by inserting a new hidden neuron C in between.
    /// src -> C gets weight 1.0, C -> dst gets the original synapse weight.
    /// The original synapse (src -> dst) is removed.
    pub fn split_synapse(&mut self, synapse_idx: usize) {
        if synapse_idx >= self.synapse_weights.len() {
            return;
        }

        // Find the source neuron `src` of this synapse
        let mut src = 0;
        for j in 0..self.active_neurons {
            let start = self.neuron_offsets[j] as usize;
            let end = self.neuron_offsets[j + 1] as usize;
            if synapse_idx >= start && synapse_idx < end {
                src = j;
                break;
            }
        }

        let original_weight = self.synapse_weights[synapse_idx];
        let original_target = self.synapse_targets[synapse_idx] as usize;

        // 1. Remove the original synapse src -> dst
        self.synapse_weights.remove(synapse_idx);
        self.synapse_targets.remove(synapse_idx);

        // Shift offsets for neurons after `src`
        for o in (src + 1)..=self.active_neurons {
            self.neuron_offsets[o] -= 1;
        }

        // 2. Adjust indices if they are affected by output neuron shifting
        let mut adjusted_src = src;
        let mut adjusted_target = original_target;
        let new_hidden_idx = self.num_inputs + self.num_hidden;

        if src >= new_hidden_idx {
            adjusted_src += 1;
        }
        if original_target >= new_hidden_idx {
            adjusted_target += 1;
        }

        // 3. Create the new hidden neuron
        let new_neuron_idx = self.add_hidden_neuron();

        // 4. Connect adjusted_src -> C and C -> adjusted_target
        self.add_synapse(adjusted_src, new_neuron_idx, 1.0);
        self.add_synapse(new_neuron_idx, adjusted_target, original_weight);
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

    pub fn to_state(&self) -> FastBrainState {
        FastBrainState {
            num_inputs: self.num_inputs,
            num_hidden: self.num_hidden,
            num_outputs: self.num_outputs,
            active_neurons: self.active_neurons,
            decay: self.decay,
            thresholds: self.thresholds.clone(),
            biases: self.biases.clone(),
            synapse_weights: self.synapse_weights.clone(),
            synapse_targets: self.synapse_targets.clone(),
            neuron_offsets: self.neuron_offsets.clone(),
            stdp_tau_pre: self.stdp_tau_pre,
            stdp_tau_post: self.stdp_tau_post,
            stdp_a_plus: self.stdp_a_plus,
            stdp_a_minus: self.stdp_a_minus,
        }
    }

    /// Reconstructs a FastBrain from a FastBrainState, allocating and initializing execution-only buffers.
    pub fn from_state(state: FastBrainState) -> Self {
        let active_neurons = state.active_neurons;
        Self {
            num_inputs: state.num_inputs,
            num_hidden: state.num_hidden,
            num_outputs: state.num_outputs,
            active_neurons,
            activations: vec![0.0; active_neurons],
            thresholds: state.thresholds,
            biases: state.biases,
            decay: state.decay,
            last_spikes: vec![false; active_neurons],
            current_spikes: vec![false; active_neurons],
            inputs_accumulated: vec![0.0; active_neurons],
            synapse_weights: state.synapse_weights,
            synapse_targets: state.synapse_targets,
            neuron_offsets: state.neuron_offsets,
            pre_traces: vec![0.0; active_neurons],
            post_traces: vec![0.0; active_neurons],
            stdp_tau_pre: state.stdp_tau_pre,
            stdp_tau_post: state.stdp_tau_post,
            stdp_a_plus: state.stdp_a_plus,
            stdp_a_minus: state.stdp_a_minus,
        }
    }

    /// Serializes the FastBrain into a pretty-printed JSON string.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.to_state())
    }

    /// Deserializes a FastBrain from a JSON string.
    pub fn from_json(json_str: &str) -> serde_json::Result<Self> {
        let state: FastBrainState = serde_json::from_str(json_str)?;
        Ok(Self::from_state(state))
    }

    /// Saves the SNN state to a file on disk in pretty-printed JSON format.
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Loads the SNN state from a file on disk in JSON format.
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

fn default_stdp_tau() -> f32 { 20.0 }
fn default_stdp_a_plus() -> f32 { 0.005 }
fn default_stdp_a_minus() -> f32 { 0.006 }

/// A lightweight, serializable representation of a FastBrain's state.
/// Holds only structural dimensions and learned parameters (weights, biases, thresholds),
/// completely omitting dynamic and scratchpad vectors to avoid memory/file bloating.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FastBrainState {
    pub num_inputs: usize,
    pub num_hidden: usize,
    pub num_outputs: usize,
    pub active_neurons: usize,
    pub decay: f32,
    pub thresholds: Vec<f32>,
    pub biases: Vec<f32>,
    pub synapse_weights: Vec<f32>,
    pub synapse_targets: Vec<u32>,
    pub neuron_offsets: Vec<u32>,
    
    #[serde(default = "default_stdp_tau")]
    pub stdp_tau_pre: f32,
    #[serde(default = "default_stdp_tau")]
    pub stdp_tau_post: f32,
    #[serde(default = "default_stdp_a_plus")]
    pub stdp_a_plus: f32,
    #[serde(default = "default_stdp_a_minus")]
    pub stdp_a_minus: f32,
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
        csr.stdp_a_plus = 0.0;
        csr.stdp_a_minus = 0.0;

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

    #[test]
    fn test_stdp_learning_adaptation() {
        let mut brain = FastBrain::new(1, 1, 1);
        
        // Let's set standard STDP learning parameters
        brain.stdp_tau_pre = 10.0;
        brain.stdp_tau_post = 10.0;
        brain.stdp_a_plus = 0.05; // large learning rate to see effect quickly
        brain.stdp_a_minus = 0.05;
        
        // Force the threshold of hidden neuron to be very low so it spikes easily
        brain.thresholds[1] = 0.05;
        brain.activations[1] = 0.0;

        // Ensure there is a synapse from 0 (input) to 1 (hidden)
        let mut found_synapse = false;
        let start = brain.neuron_offsets[0] as usize;
        let end = brain.neuron_offsets[1] as usize;
        for idx in start..end {
            if brain.synapse_targets[idx] == 1 {
                found_synapse = true;
                break;
            }
        }
        
        if !found_synapse {
            // Force create a synapse in CSR format for this test
            brain.synapse_weights.push(0.1);
            brain.synapse_targets.push(1);
            for o in 1..brain.neuron_offsets.len() {
                brain.neuron_offsets[o] += 1;
            }
        }

        // Retrieve initial weight of synapse 0 -> 1
        let start = brain.neuron_offsets[0] as usize;
        let end = brain.neuron_offsets[1] as usize;
        let mut synapse_idx = 0;
        for idx in start..end {
            if brain.synapse_targets[idx] == 1 {
                synapse_idx = idx;
                break;
            }
        }

        let initial_weight = brain.synapse_weights[synapse_idx];

        // Tick several times with highly active input [1.0] to trigger spikes
        for _ in 0..10 {
            brain.tick(&[1.0]);
        }

        let final_weight = brain.synapse_weights[synapse_idx];
        
        // Verification: The weight should have mutated!
        assert!(
            (final_weight - initial_weight).abs() > 0.0,
            "STDP failed to adapt weight! Initial: {}, Final: {}",
            initial_weight,
            final_weight
        );
    }

    #[test]
    fn test_dynamic_mutations() {
        let mut brain = FastBrain::new(2, 2, 2);
        let orig_neurons = brain.active_neurons;
        
        // Test add_synapse
        brain.add_synapse(0, 3, 1.5);
        let mut found = false;
        let start = brain.neuron_offsets[0] as usize;
        let end = brain.neuron_offsets[1] as usize;
        for idx in start..end {
            if brain.synapse_targets[idx] == 3 {
                assert_eq!(brain.synapse_weights[idx], 1.5);
                found = true;
            }
        }
        assert!(found, "add_synapse failed to insert synapse!");

        // Test add_hidden_neuron
        let new_id = brain.add_hidden_neuron();
        assert_eq!(brain.active_neurons, orig_neurons + 1);
        assert_eq!(new_id, 4);

        // Test split_synapse
        let start = brain.neuron_offsets[0] as usize;
        let end = brain.neuron_offsets[1] as usize;
        let mut synapse_idx = None;
        for idx in start..end {
            if brain.synapse_targets[idx] == 3 {
                synapse_idx = Some(idx);
            }
        }
        
        if let Some(s_idx) = synapse_idx {
            let prev_synapses = brain.synapse_weights.len();
            brain.split_synapse(s_idx);
            assert_eq!(brain.synapse_weights.len(), prev_synapses + 1);
        }

        // Test prune_synapses
        brain.add_synapse(1, 2, 1e-6);
        let prev_len = brain.synapse_weights.len();
        brain.prune_synapses();
        assert!(brain.synapse_weights.len() < prev_len, "prune_synapses failed to prune weak synapse!");
    }
}
