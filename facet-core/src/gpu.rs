pub mod matmul;

use std::sync::Arc;
use vulkano::{
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
};

#[derive(thiserror::Error, Debug)]
pub enum GpuNdArrayError {
    #[error("NdArray native error {0}")]
    NdArrayError(crate::ndarray::NdArrayError),
    #[error("Could not load shader")]
    NoShader,
    #[error("Could not load GPU executor")]
    NoExecutor,
}

#[allow(unused)]
pub struct GpuExecutor {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

lazy_static::lazy_static! {
    pub static ref EXECUTOR: Option<GpuExecutor> = init();
}

pub fn init() -> Option<GpuExecutor> {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).ok()?;

    let physical = PhysicalDevice::enumerate(&instance)
        .max_by_key(|device| device.limits().max_compute_shared_memory_size())?;

    let queue_family = physical.queue_families().find(|&q| q.supports_compute())?;

    let (device, mut queues) = Device::new(
        physical,
        &Features {
            shader_float64: true,
            ..Default::default()
        },
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
        [(queue_family, 0.5)].iter().cloned(),
    )
    .ok()?;

    let queue = queues.next().unwrap();

    let exc = GpuExecutor {
        instance,
        queue,
        device,
    };
    Some(exc)
}
