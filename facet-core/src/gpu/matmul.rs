//! compute shaders
//!

use super::{GpuNdArrayError, EXECUTOR, MATMUL};

use rayon::prelude::*;
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    descriptor::PipelineLayoutAbstract,
};

// naive impl
// TODO optimize
// maybe something like this? https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm6.htm
vulkano_shaders::shader! {
    ty: "compute",
    src: r#"
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Data_a { double A[]; };
layout(set = 0, binding = 1) readonly  buffer Data_b { double B[]; };
layout(set = 0, binding = 2) writeonly buffer Data_c { double C[]; };

layout(push_constant) uniform Shape {
    uint N;
    uint M;
    uint P;
};

void main()
{
    uint i = gl_GlobalInvocationID.x; // [0..n)
    uint j = gl_GlobalInvocationID.y; // [0..p)

    double value = 0.0;
    for(uint k = 0; k < M; k++)
    {
        double a = A[i * M + k];
        double b = B[k * P + j];
        value += a * b;
    }

    C[i * P + j] = value;
}"#
}

pub fn matmul_f64_impl<'a>(
    [n, m, p]: [u32; 3],
    values0: &'a [f64],
    values1: &'a [f64],
    out: &mut [f64],
) -> Result<(), GpuNdArrayError> {
    debug_assert_eq!((n as usize * m as usize), values0.len());
    debug_assert_eq!((p as usize * m as usize), values1.len());
    debug_assert_eq!(out.len(), n as usize * p as usize);

    let shape = [n, m, p];

    let shader = match MATMUL.as_ref() {
        Some(shader) => shader,
        None => return Err(GpuNdArrayError::NoShader),
    };
    let exc = EXECUTOR.as_ref().unwrap();
    let device = exc.device.clone();
    let compute_pipeline = Arc::new(
        vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
            None,
        )
        .expect("failed to create compute pipeline"),
    );

    dbg!(
        values0.len() + values1.len() + out.len(),
        out.len(),
        exc.buffer_pool_f64.capacity()
    );
    // force larger allocations
    exc.buffer_pool_f64
        .reserve((values0.len() + values1.len() + out.len()).max(256 * 1024 * 1024)) // 256 MiB
        .unwrap();
    let a_buffer = exc
        .buffer_pool_f64
        .chunk(values0.iter().cloned())
        .expect("buffer a");
    let b_buffer = exc
        .buffer_pool_f64
        .chunk(values1.iter().cloned())
        .expect("buffer b");

    let c_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        true,
        (0..out.len()).map(|_| 0.0f64),
    )
    .expect("failed to create buffer c");

    // Descriptor sets
    let descriptor = vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
        compute_pipeline
            .layout()
            .descriptor_set_layout(0)
            .unwrap()
            .clone(),
    )
    .add_buffer(a_buffer)
    .expect("a buffer")
    .add_buffer(b_buffer)
    .expect("b buffer")
    .add_buffer(c_buffer.clone())
    .expect("c buffer")
    .build()
    .unwrap();

    // Dispatch
    let mut builder =
        vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), exc.queue.family())
            .unwrap();
    builder
        .dispatch([n, p, 1], compute_pipeline.clone(), descriptor, shape)
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let finish = vulkano::sync::now(device.clone());
    let finish = finish
        .then_execute(exc.queue.clone(), command_buffer)
        .unwrap();
    let finish = finish
        .then_signal_fence_and_flush()
        .expect("failed to flush");
    finish.wait(None).expect("compute shader execution failed");

    // gpu finished processing, copy the result
    let content = c_buffer.read().unwrap();
    out.par_iter_mut().enumerate().for_each(|(i, b)| {
        *b = content[i];
    });

    rayon::spawn(move || {
        let mut finish = finish;
        finish.cleanup_finished();
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use rand::Rng;

    #[test]
    fn test_correctness() {
        const N: u32 = 4;
        const M: u32 = 7;
        const P: u32 = 3;

        let mut rng = rand::thread_rng();
        let a =
            NdArray::new_with_values([N, M], (0..N * M).map(|_| rng.gen_range(0., 10.)).collect())
                .unwrap();

        let b = NdArray::new_with_values(
            &[M, P][..],
            (0..M * P).map(|_| rng.gen_range(0., 10.)).collect(),
        )
        .unwrap();

        let mut c_gpu = NdArray::new(N * P);
        matmul_f64_impl([N, M, P], a.as_slice(), b.as_slice(), c_gpu.as_mut_slice()).unwrap();

        let mut c_cpu = NdArray::new(N * P);
        crate::ndarray::matrix::matmul_impl(
            [N, M, P],
            a.as_slice(),
            b.as_slice(),
            c_cpu.as_mut_slice(),
        )
        .unwrap();

        dbg!(&c_gpu, &c_cpu);

        for (a, b) in c_gpu.as_slice().iter().zip(c_cpu.as_slice().iter()) {
            assert!((a - b).abs() < 0.000001)
        }
    }
}
