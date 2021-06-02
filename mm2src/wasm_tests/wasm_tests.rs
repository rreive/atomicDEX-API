pub mod command;
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-integration-test")))]
pub mod instance_interface;
#[cfg(all(target_arch = "wasm32", feature = "wasm-integration-test"))]
pub mod instance_runner;
