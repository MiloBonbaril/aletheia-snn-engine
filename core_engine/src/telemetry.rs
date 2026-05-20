use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// TelemetryHub handles asynchronous tracking of spiking activity.
/// It uses lock-free atomic variables or channels to pass data to the twitch_dashboard.
#[derive(Clone)]
pub struct TelemetryHub {
    last_spikes: Arc<AtomicU64>,
}

impl TelemetryHub {
    /// Creates a new TelemetryHub initialized with 0 active spikes.
    pub fn new() -> Self {
        Self {
            last_spikes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record the active neuron bitmask.
    /// FastBrain or the PyO3 wrapper can call this to push telemetry lock-free.
    pub fn record_spikes(&self, mask: u64) {
        self.last_spikes.store(mask, Ordering::Relaxed);
    }

    /// Get the last recorded spikes bitmask.
    pub fn get_last_spikes(&self) -> u64 {
        self.last_spikes.load(Ordering::Relaxed)
    }

    /// Starts a placeholder background telemetry loop to simulate stream streaming.
    pub fn start_background_loop(&self) {
        let spikes_clone = self.last_spikes.clone();
        std::thread::spawn(move || {
            // In the real system, this thread sends WebSocket messages to the Twitch Dashboard
            let mut last_value = 0;
            loop {
                let mask = spikes_clone.load(Ordering::Relaxed);
                if mask != last_value {
                    // Spikes changed, we would send a frame to WebGL dashboard
                    last_value = mask;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }
}
