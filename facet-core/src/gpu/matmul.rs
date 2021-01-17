//! compute shaders
//!
#[cfg(test)]
mod tests;

use super::{GpuNdArrayError, EXECUTOR, MATMUL};

use rayon::prelude::*;
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    descriptor::PipelineLayoutAbstract,
};

pub const LOCAL_SIZE_X: u32 = 32;
pub const LOCAL_SIZE_Y: u32 = 16;
/// Number of rows in the left matrix to process at a time
pub const ROW_SPLIT_THRESHOLD: u32 = 1024;

// naive impl
// TODO optimize
// maybe something like this? https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm6.htm
vulkano_shaders::shader! {
    ty: "compute",
    src: r#"
#version 450

layout(local_size_x = 32, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Data_a { float A[]; };
layout(set = 0, binding = 1) readonly  buffer Data_b { float B[]; };
layout(set = 0, binding = 2) writeonly buffer Data_c { float C[]; };

layout(push_constant) uniform Shape {
    uint N;
    uint M;
    uint P;
};

void main()
{
    uint i = gl_GlobalInvocationID.x; // [0..n)
    uint j = gl_GlobalInvocationID.y; // [0..p)

    float value = 0.0;
    for(uint k = 0; k < M; k++)
    {
        float a = A[i * M + k];
        float b = B[k * P + j];
        value += a * b;
    }

    C[i * P + j] = value;
}"#
}

lazy_static::lazy_static! {
    pub static ref PIPELINE: Arc<vulkano::pipeline::ComputePipeline<PipelineLayout<Layout>>> = {
    let exc = EXECUTOR.as_ref().unwrap();
    let device = exc.device.clone();
    let shader = match MATMUL.as_ref() {
        Some(shader) => shader,
        None => panic!("{}",GpuNdArrayError::NoShader),
    };
    Arc::new(
        vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
            None,
        )
        .expect("failed to create compute pipeline"),
    )
    };
}

pub fn matmul_f32_impl<'a>(
    [n, m, p]: [u32; 3],
    values0: &'a [f32],
    values1: &'a [f32],
    out: &mut [f32],
) -> Result<(), GpuNdArrayError> {
    assert!(n as usize * m as usize <= values0.len());
    assert!(p as usize * m as usize <= values1.len());
    assert!(n as usize * p as usize <= out.len());

    let exc = match EXECUTOR.as_ref() {
        Some(e) => e,
        None => return Err(GpuNdArrayError::NoExecutor),
    };
    let device = exc.device.clone();
    let compute_pipeline = PIPELINE.clone();

    let res = if n > ROW_SPLIT_THRESHOLD {
        // iterate over some of the rows at a time
        let device = device.clone();
        out.par_chunks_mut(p as usize * ROW_SPLIT_THRESHOLD as usize)
            .enumerate()
            .try_for_each(move |(subi, submatrix)| {
                let offset = subi * ROW_SPLIT_THRESHOLD as usize;
                let n = submatrix.len() / p as usize; // 1..ROW_SPLIT
                debug_assert!(n >= 1);
                debug_assert!(values0[offset * m as usize..].len() >= n * m as usize);
                _matmul_f32_impl(
                    exc,
                    device.clone(),
                    compute_pipeline.clone(),
                    [n as u32, m, p],
                    &values0[offset * m as usize..],
                    values1,
                    submatrix,
                )
            })
    } else {
        _matmul_f32_impl(
            exc,
            device.clone(),
            compute_pipeline,
            [n, m, p],
            values0,
            values1,
            out,
        )
    };

    // we should periodically clean up the gpu resources, for now let's do that in every call
    let mut finish = vulkano::sync::now(device);
    finish.cleanup_finished();

    res
}

pub fn _matmul_f32_impl<'a>(
    exc: &super::GpuExecutor,
    device: Arc<Device>,
    compute_pipeline: Arc<vulkano::pipeline::ComputePipeline<PipelineLayout<Layout>>>,
    // matmul params
    [n, m, p]: [u32; 3],
    values0: &'a [f32],
    values1: &'a [f32],
    out: &mut [f32],
) -> Result<(), GpuNdArrayError> {
    let shape = [n, m, p];

    let ((a_buffer, b_buffer), c_buffer) = rayon::join(
        || {
            rayon::join(
                || matrix_buffer(device.clone(), false, values0.iter().cloned()),
                || matrix_buffer(device.clone(), false, values1.iter().cloned()),
            )
        },
        || matrix_buffer(device.clone(), true, (0..out.len()).map(|_| 0.0f32)),
    );

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
    let workgroups = [n / LOCAL_SIZE_X, p / LOCAL_SIZE_Y, 1];
    builder
        .dispatch(workgroups, compute_pipeline, descriptor, shape)
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let finish = vulkano::sync::now(device)
        .then_execute(exc.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .expect("failed to flush");

    // buffers can be reused, ensure 0 initial output value
    out.par_iter_mut().for_each(|x| *x = 0.0);

    // process the remaning columns on the cpu while we await the gpu execution
    // note that the last block is calculated twice, the auther deems this ok for now

    // last columns
    let remaining_n = n % LOCAL_SIZE_X;
    let offset_n = (n - remaining_n) as usize;
    (0..p).for_each(|j| {
        let j = j as usize;
        for i in 0..remaining_n {
            let i = i as usize + offset_n;
            let mut val = 0.0;
            for k in 0..m {
                let k = k as usize;
                val += at(values0, i, m as usize, k) * at(values1, k, p as usize, j);
            }
            out[i * p as usize + j] = val;
        }
    });
    // last rows
    let remaining_p = p % LOCAL_SIZE_Y;
    let offset_p = (p - remaining_p) as usize;
    (0..n).for_each(|i| {
        let i = i as usize;
        for j in 0..remaining_p {
            let j = j as usize + offset_p;
            let mut val = 0.0;
            for k in 0..m {
                let k = k as usize;
                val += at(values0, i, m as usize, k) * at(values1, k, p as usize, j);
            }
            out[i * p as usize + j] = val;
        }
    });

    finish.wait(None).expect("compute shader execution failed");

    // gpu finished processing, copy the result
    let content = c_buffer.read().unwrap();
    out.par_chunks_mut(p as usize)
        .enumerate()
        // only set the out values we haven't touched in the cpu-computations
        .for_each(|(i, b)| {
            let offset = i * p as usize;
            for (i, v) in b.iter_mut().enumerate() {
                // use add as we might touch values skipped by the gpu
                // these values will be set to 0.
                *v += content[offset + i];
            }
        });

    Ok(())
}

#[inline(always)]
fn at(values: &[f32], i: usize, cols: usize, j: usize) -> f32 {
    unsafe { *values.as_ptr().add(i * cols + j) }
}

#[inline]
fn matrix_buffer(
    device: Arc<Device>,
    host_cached: bool,
    data: impl Iterator<Item = f32> + std::iter::ExactSizeIterator,
) -> Arc<CpuAccessibleBuffer<[f32]>> {
    // use CPU accessible buffers (shared buffers) because I found that dedicated GPU memory runs
    // out fast
    CpuAccessibleBuffer::from_iter(device, BufferUsage::all(), host_cached, data).unwrap()
}
