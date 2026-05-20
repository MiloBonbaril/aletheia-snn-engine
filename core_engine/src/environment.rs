/// A game-agnostic and device-agnostic environment interface.
/// Decouples the spiking neural network engine from any specific game, emulator, or simulator.
pub trait Environment {
    /// Returns the shape of the environment: (inputs/sensors, outputs/actuators).
    fn shape(&self) -> (usize, usize);

    /// Observes the current state of the environment, returning a vector of sensor measurements.
    fn get_state(&self) -> Vec<f32>;

    /// Applies the network's continuous action outputs and advances the simulation by one step/frame.
    /// Returns a tuple of: (new_state, reward, is_done).
    fn step(&mut self, actions: &[f32]) -> (Vec<f32>, f32, bool);

    /// Resets the environment state to its initial conditions for a new episode or generation.
    fn reset(&mut self);
}
