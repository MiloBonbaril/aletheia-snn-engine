use pyo3::prelude::*;
use core_engine::brain::FastBrain as CoreFastBrain;
use core_engine::telemetry::TelemetryHub;

/// PyO3 wrapped FastBrain class for Gymnasium interaction.
#[pyclass]
pub struct FastBrain {
    inner: CoreFastBrain,
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
            inner: CoreFastBrain::new(num_inputs, num_hidden, num_outputs),
            telemetry,
        }
    }

    /// Ingests sensor input observations from Python, triggers SNN propagation
    /// to compute the continuous motor actions, pushes the spikes bitmask
    /// to telemetry, and returns the actions.
    pub fn tick(&mut self, inputs: Vec<f32>) -> PyResult<Vec<f32>> {
        if inputs.len() != self.inner.num_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("FastBrain requires exactly {} inputs, got {}", self.inner.num_inputs, inputs.len())
            ));
        }

        // 1. Python says: "Here is the state of the environment (floats)"
        // 2. PyO3 passes it directly to core_engine in RAM
        // 3. FastBrain calculates spike propagation and computes actions
        let actions = self.inner.tick(&inputs);

        // 4. FastBrain pushes the active neurons bitmask to the telemetry hub
        let spikes_bitmask = self.inner.get_last_spikes();
        self.telemetry.record_spikes(spikes_bitmask);

        // 5. Python retrieves the 4 actions to drive the robot
        Ok(actions.to_vec())
    }

    /// Retrieve the current active spikes bitmask.
    pub fn get_last_spikes(&self) -> u64 {
        self.inner.get_last_spikes()
    }

    /// Retrieve the spikes bitmask recorded by the telemetry hub.
    pub fn get_telemetry_spikes(&self) -> u64 {
        self.telemetry.get_last_spikes()
    }
}

/// The Aletheia SNN Python module entry point.
#[pymodule]
fn python_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastBrain>()?;
    Ok(())
}
