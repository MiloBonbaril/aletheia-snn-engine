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
    /// Create a new FastBrain instance.
    #[new]
    pub fn new() -> Self {
        let telemetry = TelemetryHub::new();
        // Start the background telemetry loop (Simulated telemetry server thread)
        telemetry.start_background_loop();
        
        Self {
            inner: CoreFastBrain::new(),
            telemetry,
        }
    }

    /// Ingests 24 sensor input observations from Python, triggers SNN propagation
    /// to compute the 4 continuous motor actions, pushes the spikes bitmask
    /// to telemetry, and returns the actions.
    pub fn tick(&mut self, inputs: Vec<f32>) -> PyResult<Vec<f32>> {
        if inputs.len() != 24 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("FastBrain requires exactly 24 inputs, got {}", inputs.len())
            ));
        }

        // 1. Python says: "Here is the state of BipedalWalker (24 floats)"
        // 2. PyO3 passes it directly to core_engine in RAM
        // 3. FastBrain calculates spike propagation and computes the 4 actions
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
