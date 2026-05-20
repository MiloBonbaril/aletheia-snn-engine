// CSR (Compressed Sparse Row) matrix structure and SNN inference engine

/// FastBrain represents a zero-allocation, high-performance Spiking Neural Network.
/// It consists of:
/// - 24 Input neurons (0..24) driven by Gymnasium observations.
/// - 36 Hidden neurons (24..60) with leaky integration, recurrent connections, and spike generation.
/// - 4 Output neurons (60..64) whose membrane potentials map to the 4 motor controls.
///
/// This design uses static/flat arrays, achieving 100% zero-allocation in the inference loop.
pub struct FastBrain {
    /// Membrane potentials of the 64 neurons.
    pub potentials: [f32; 64],
    /// Thresholds for firing for each neuron.
    pub thresholds: [f32; 64],
    /// Fully connected weight matrix (64 x 64 = 4096 elements).
    /// weight[src * 64 + dst] represents the synaptic connection from src to dst.
    pub weights: [f32; 4096],
    /// Leak decay rate (e.g., 0.9 means potentials decay by 10% each tick).
    pub decay: f32,
    /// Bitmask of neurons that spiked in the current/last tick.
    pub last_spikes: u64,
}

impl FastBrain {
    /// Creates a new FastBrain with a deterministic connection structure.
    pub fn new() -> Self {
        let potentials = [0.0; 64];
        let mut thresholds = [1.0; 64];
        let mut weights = [0.0; 4096];
        let decay = 0.85;

        // Set default thresholds for hidden and output neurons
        for i in 24..64 {
            thresholds[i] = 1.0;
        }

        // Initialize weights with a simple, interesting pattern.
        // Let's connect input neurons to hidden neurons, and hidden neurons to output neurons.
        // We also add some recurrent hidden-to-hidden connections.
        // We use a simple LCG pseudo-random number generator to seed weights deterministically.
        let mut seed: u32 = 42;
        let mut next_random = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed / 65536) % 32768) as f32 / 32767.0 * 2.0 - 1.0 // Range [-1.0, 1.0]
        };

        // Inputs (0..24) to Hidden (24..60)
        for src in 0..24 {
            for dst in 24..60 {
                // Sparsely connect: 40% connection probability
                if next_random() > -0.2 {
                    weights[src * 64 + dst] = next_random() * 0.5;
                }
            }
        }

        // Recurrent Hidden (24..60) to Hidden (24..60)
        for src in 24..60 {
            for dst in 24..60 {
                if src != dst && next_random() > 0.6 {
                    weights[src * 64 + dst] = next_random() * 0.2;
                }
            }
        }

        // Hidden (24..60) to Output (60..64)
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

    /// Performs one simulation step (tick) of the SNN.
    ///
    /// - `inputs`: 24 float inputs representing the environment state.
    /// - Returns: 4 float actions mapped to [-1.0, 1.0] via hyperbolic tangent of the output potentials.
    pub fn tick(&mut self, inputs: &[f32]) -> [f32; 4] {
        let mut current_spikes: u64 = 0;

        // 1. Process Input Neurons (0..24)
        // Set input potentials directly from inputs. If positive, they spike.
        for i in 0..24 {
            let val = inputs.get(i).copied().unwrap_or(0.0);
            self.potentials[i] = val;
            if val > 0.0 {
                current_spikes |= 1 << i;
            }
        }

        // 2. Process Hidden Neurons (24..60)
        // Hidden neurons receive inputs from input spikes and recurrent spikes from the last tick.
        for dst in 24..60 {
            let mut current = 0.0;

            // Direct input driving
            for src in 0..24 {
                current += self.weights[src * 64 + dst] * self.potentials[src];
            }

            // Recurrent hidden spikes from last tick
            for src in 24..60 {
                if (self.last_spikes & (1 << src)) != 0 {
                    current += self.weights[src * 64 + dst];
                }
            }

            // Integrate leaky potential
            self.potentials[dst] = self.potentials[dst] * self.decay + current;

            // Check spike threshold
            if self.potentials[dst] >= self.thresholds[dst] {
                current_spikes |= 1 << dst;
                self.potentials[dst] = 0.0; // Reset
            }
        }

        // 3. Process Output Neurons (60..64)
        // Output neurons receive inputs from currently spiking hidden neurons.
        for dst in 60..64 {
            let mut current = 0.0;
            for src in 24..60 {
                if (current_spikes & (1 << src)) != 0 {
                    current += self.weights[src * 64 + dst];
                }
            }

            // Integrate leaky potential
            self.potentials[dst] = self.potentials[dst] * self.decay + current;

            // Check spike threshold
            if self.potentials[dst] >= self.thresholds[dst] {
                current_spikes |= 1 << dst;
                self.potentials[dst] = 0.0; // Reset
            }
        }

        // Save current spikes for recurrent step next tick
        self.last_spikes = current_spikes;

        // Map the output membrane potentials to 4 continuous motor actions in [-1.0, 1.0]
        let mut actions = [0.0; 4];
        for i in 0..4 {
            // Apply a smooth activation function (tanh) to map potentials to [-1.0, 1.0]
            actions[i] = self.potentials[60 + i].tanh();
        }

        actions
    }

    /// Exposes the current spike bitmask.
    pub fn get_last_spikes(&self) -> u64 {
        self.last_spikes
    }
}
