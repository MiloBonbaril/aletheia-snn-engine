// High-performance CUDA SNN propagation kernel for batch simulation

extern "C" __global__ void compute_snn_kernel(
    const int num_neurons,
    const int num_clones,
    const int num_synapses,
    float* membrane_potentials,  // size: num_clones * num_neurons (VRAM persist)
    int* spike_out,              // size: num_clones * num_neurons (output spikes)
    const int* spike_in,         // size: num_clones * num_neurons (input spikes)
    const float* weights,        // size: num_clones * num_synapses (perturbed weights)
    const int* synapse_targets,  // size: num_synapses (shared CSR targets)
    const int* neuron_offsets,   // size: num_neurons + 1 (shared CSR offsets)
    const float* thresholds,     // size: num_neurons (shared thresholds)
    const float beta             // Leak decay rate
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_clones * num_neurons;
    
    if (global_thread_id < total_threads) {
        int clone_id = global_thread_id / num_neurons;
        int neuron_id = global_thread_id % num_neurons;
        
        float incoming_current = 0.0f;
        int start_idx = neuron_offsets[neuron_id];
        int end_idx   = neuron_offsets[neuron_id + 1];
        
        // Accumulate currents from all active synapses originating from spiking neurons
        for (int i = start_idx; i < end_idx; ++i) {
            int target_id = synapse_targets[i];
            
            // Check if the source neuron spiked in the previous step for this specific clone
            if (spike_in[clone_id * num_neurons + target_id] == 1) {
                // Fetch the unique synaptic weight of this clone
                incoming_current += weights[clone_id * num_synapses + i];
            }
        }
        
        // Potential update: Leak decay + incoming current
        int potential_idx = clone_id * num_neurons + neuron_id;
        float v = membrane_potentials[potential_idx] * beta + incoming_current;
        
        float thresh = thresholds[neuron_id];
        
        // Fire & Reset (Leaky Integrate-and-Fire mechanics)
        if (v >= thresh) {
            spike_out[potential_idx] = 1;
            membrane_potentials[potential_idx] = 0.0f; // Reset membrane potential to 0.0
        } else {
            spike_out[potential_idx] = 0;
            membrane_potentials[potential_idx] = v;    // Retain integrated potential
        }
    }
}

extern "C" __global__ void evaluate_phantom_mutants_kernel(
    const int* dims,                    // size: 10
    const float* inputs_buffer,         // size: capacity * num_inputs
    const float* historic_actions,      // size: capacity * num_outputs
    const float* rewards,               // size: capacity
    const float* weights,               // size: num_clones * num_synapses
    const int* topology_data,           // size: num_neurons + 1 + num_synapses (offsets + targets)
    const float* neuron_params,         // size: 2 * num_neurons (thresholds + biases)
    float* fitness_out,                  // size: num_clones
    float* workspace                    // size: 4 * num_clones * num_neurons
) {
    int clone_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int num_inputs   = dims[0];
    int num_hidden   = dims[1];
    int num_outputs  = dims[2];
    int num_neurons  = dims[3];
    int num_clones   = dims[4];
    int num_synapses = dims[5];
    int history_len  = dims[6];
    int oldest_idx   = dims[7];
    int capacity     = dims[8];
    float decay      = __int_as_float(dims[9]);

    if (clone_id >= num_clones) return;

    // Workspace offsets
    float* workspace_activations = workspace;
    float* workspace_prev_spikes = workspace + num_clones * num_neurons;
    float* workspace_current_spikes = workspace + 2 * num_clones * num_neurons;
    float* workspace_inputs_accumulated = workspace + 3 * num_clones * num_neurons;

    // Topology offsets
    const int* neuron_offsets = topology_data;
    const int* synapse_targets = topology_data + num_neurons + 1;

    // Neuron params offsets
    const float* thresholds = neuron_params;
    const float* biases = neuron_params + num_neurons;

    // 1. Initialize clone state to 0
    for (int n = 0; n < num_neurons; ++n) {
        int idx = clone_id * num_neurons + n;
        workspace_activations[idx] = 0.0f;
        workspace_prev_spikes[idx] = 0.0f;
        workspace_current_spikes[idx] = 0.0f;
        workspace_inputs_accumulated[idx] = 0.0f;
    }

    float fitness = 0.0f;
    int hidden_start = num_inputs;
    int hidden_end = num_inputs + num_hidden;
    int output_start = hidden_end;
    int output_end = num_neurons;

    // 2. Loop over the history sequence sequentially
    for (int t = 0; t < history_len; ++t) {
        int frame_idx = (oldest_idx + t) % capacity;

        // Reset inputs_accumulated and current_spikes for this tick
        for (int n = 0; n < num_neurons; ++n) {
            int idx = clone_id * num_neurons + n;
            workspace_inputs_accumulated[idx] = 0.0f;
            workspace_current_spikes[idx] = 0.0f;
        }

        // Process input observations
        for (int i = 0; i < num_inputs; ++i) {
            float val = inputs_buffer[frame_idx * num_inputs + i];
            int idx = clone_id * num_neurons + i;
            workspace_activations[idx] = val;
            if (val > 0.0f) {
                workspace_current_spikes[idx] = 1.0f;
            }
        }

        // Propagate input activations to hidden neurons
        for (int src = 0; src < num_inputs; ++src) {
            float act = workspace_activations[clone_id * num_neurons + src];
            if (act != 0.0f) {
                int start = neuron_offsets[src];
                int end = neuron_offsets[src + 1];
                for (int idx = start; idx < end; ++idx) {
                    int target = synapse_targets[idx];
                    if (target >= hidden_start && target < hidden_end) {
                        workspace_inputs_accumulated[clone_id * num_neurons + target] += act * weights[clone_id * num_synapses + idx];
                    }
                }
            }
        }

        // Propagate recurrent spikes from the previous step
        for (int src = hidden_start; src < hidden_end; ++src) {
            if (workspace_prev_spikes[clone_id * num_neurons + src] == 1.0f) {
                int start = neuron_offsets[src];
                int end = neuron_offsets[src + 1];
                for (int idx = start; idx < end; ++idx) {
                    int target = synapse_targets[idx];
                    if (target >= hidden_start && target < hidden_end) {
                        workspace_inputs_accumulated[clone_id * num_neurons + target] += weights[clone_id * num_synapses + idx];
                    }
                }
            }
        }

        // Integrate leaky membrane potentials for hidden neurons and fire
        for (int dst = hidden_start; dst < hidden_end; ++dst) {
            int idx = clone_id * num_neurons + dst;
            float current = workspace_inputs_accumulated[idx] + biases[dst];
            float v = workspace_activations[idx] * decay + current;
            float thresh = thresholds[dst];

            if (v >= thresh) {
                workspace_current_spikes[idx] = 1.0f;
                workspace_activations[idx] = 0.0f;
            } else {
                workspace_current_spikes[idx] = 0.0f;
                workspace_activations[idx] = v;
            }
        }

        // Propagate hidden spikes to output neurons
        for (int src = hidden_start; src < hidden_end; ++src) {
            if (workspace_current_spikes[clone_id * num_neurons + src] == 1.0f) {
                int start = neuron_offsets[src];
                int end = neuron_offsets[src + 1];
                for (int idx = start; idx < end; ++idx) {
                    int target = synapse_targets[idx];
                    if (target >= output_start && target < output_end) {
                        workspace_inputs_accumulated[clone_id * num_neurons + target] += weights[clone_id * num_synapses + idx];
                    }
                }
            }
        }

        // Integrate leaky membrane potentials for output neurons and fire
        for (int dst = output_start; dst < output_end; ++dst) {
            int idx = clone_id * num_neurons + dst;
            float current = workspace_inputs_accumulated[idx] + biases[dst];
            float v = workspace_activations[idx] * decay + current;
            float thresh = thresholds[dst];

            if (v >= thresh) {
                workspace_current_spikes[idx] = 1.0f;
                workspace_activations[idx] = 0.0f;
            } else {
                workspace_current_spikes[idx] = 0.0f;
                workspace_activations[idx] = v;
            }
        }

        // Copy current spikes to prev spikes for the next step
        for (int n = 0; n < num_neurons; ++n) {
            int idx = clone_id * num_neurons + n;
            workspace_prev_spikes[idx] = workspace_current_spikes[idx];
        }

        // Compute action similarity for this frame and accumulate fitness
        float frame_dot = 0.0f;
        for (int o = 0; o < num_outputs; ++o) {
            float v_out = workspace_activations[clone_id * num_neurons + output_start + o];
            float a_clone = tanhf(v_out);
            float a_hist = historic_actions[frame_idx * num_outputs + o];
            frame_dot += a_clone * a_hist;
        }
        fitness += rewards[frame_idx] * frame_dot;
    }

    fitness_out[clone_id] = fitness;
}

extern "C" __global__ void find_champion_kernel(
    const float* fitnesses,
    const int num_clones,
    float* max_fitness_out,
    int* champion_id_out
) {
    __shared__ float sdata_val[1024];
    __shared__ int sdata_idx[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = -1e20f;
    int best_idx = -1;
    
    if (idx < num_clones) {
        val = fitnesses[idx];
        best_idx = idx;
    }
    
    sdata_val[tid] = val;
    sdata_idx[tid] = best_idx;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata_val[tid + s] > sdata_val[tid]) {
                sdata_val[tid] = sdata_val[tid + s];
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        *max_fitness_out = sdata_val[0];
        *champion_id_out = sdata_idx[0];
    }
}

