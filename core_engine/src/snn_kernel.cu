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
