/// PhantomReplayBuffer holds a contiguous flat memory layout (circular ring buffer)
/// of sensor inputs, taken actions, and received rewards from the actual game execution.
/// Designed for high-speed transfer to GPU VRAM (Zero-Copy Pinned Memory structure).
#[derive(Clone, Debug)]
pub struct PhantomReplayBuffer {
    /// Flat buffer of input sensor observations. Size: capacity * num_inputs
    pub inputs: Vec<f32>,
    /// Flat buffer of output actuator actions. Size: capacity * num_outputs
    pub actions: Vec<f32>,
    /// Buffer of rewards for each frame. Size: capacity
    pub rewards: Vec<f32>,
    
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub capacity: usize,
    pub write_idx: usize,
    pub count: usize,
}

impl PhantomReplayBuffer {
    /// Instantiates a new PhantomReplayBuffer with pre-allocated flat vectors.
    pub fn new(num_inputs: usize, num_outputs: usize, capacity: usize) -> Self {
        Self {
            inputs: vec![0.0; capacity * num_inputs],
            actions: vec![0.0; capacity * num_outputs],
            rewards: vec![0.0; capacity],
            num_inputs,
            num_outputs,
            capacity,
            write_idx: 0,
            count: 0,
        }
    }

    /// Adds a single game frame (sensor state, actuator outputs, and immediate feedback) to the buffer.
    /// Operates entirely in place with 0% dynamic heap allocation overhead.
    pub fn add_frame(&mut self, input: &[f32], action: &[f32], reward: f32) {
        if input.len() != self.num_inputs || action.len() != self.num_outputs {
            return; // Dimension mismatch safety valve
        }

        let idx = self.write_idx;

        // Ingest inputs
        let input_start = idx * self.num_inputs;
        self.inputs[input_start..(input_start + self.num_inputs)].copy_from_slice(input);

        // Ingest actions
        let action_start = idx * self.num_outputs;
        self.actions[action_start..(action_start + self.num_outputs)].copy_from_slice(action);

        // Ingest reward
        self.rewards[idx] = reward;

        // Advance circular index
        self.write_idx = (idx + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Resets the buffer's writing index and counter, zeroing out pre-allocated memory.
    pub fn reset(&mut self) {
        self.inputs.fill(0.0);
        self.actions.fill(0.0);
        self.rewards.fill(0.0);
        self.write_idx = 0;
        self.count = 0;
    }
}
