//! compute shaders
//!

use super::{GpuNdArrayError, EXECUTOR, MATMUL};

use rayon::prelude::*;
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    descriptor::PipelineLayoutAbstract,
};

pub const LOCAL_SIZE_X: u32 = 8;
pub const LOCAL_SIZE_Y: u32 = 8;

// naive impl
// TODO optimize
// maybe something like this? https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm6.htm
vulkano_shaders::shader! {
    ty: "compute",
    src: r#"
#version 450

// tile size
#define TS 8

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Data_a { double A[]; };
layout(set = 0, binding = 1) readonly  buffer Data_b { double B[]; };
layout(set = 0, binding = 2) writeonly buffer Data_c { double C[]; };

layout(push_constant) uniform Shape {
    uint N;
    uint M;
    uint P;
};

// 8 by 8 tiles
shared double Asub[TS][TS];
shared double Bsub[TS][TS];

void main()
{
    uint col = gl_LocalInvocationID.x; // [0..TS)
    uint row = gl_LocalInvocationID.y; // [0..TS)
    uint globalCol = TS * gl_WorkGroupID.x + col; // [0..n)
    uint globalRow = TS * gl_WorkGroupID.y + row; // [0..p)

    double value = 0.0;
    const uint numTiles = P/TS;
    for(uint t = 0; t < numTiles; ++t)
    {
        const uint tiledRow = TS*t + row;
        const uint tiledCol = TS*t + col;

        // load the local tile
        Asub[col][row] = A[tiledCol * N + globalRow];
        Bsub[col][row] = B[globalCol * P + tiledRow];

        // wait for Asub and Bsub to be completely filled
        barrier();

        // calculate the value using the pre-loaded tile
        for (uint k=0; k < TS; ++k)
        {
            value += Asub[k][row] * Bsub[col][k];
        }

        // sync before loading the next tile  
        barrier();
    }

    C[globalCol * P + globalRow] = value;
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

    let ((a_buffer, b_buffer), c_buffer) = rayon::join(
        || {
            rayon::join(
                || {
                    exc.buffer_pool_f64
                        .chunk(values0.iter().cloned())
                        .expect("buffer a")
                },
                || {
                    exc.buffer_pool_f64
                        .chunk(values1.iter().cloned())
                        .expect("buffer b")
                },
            )
        },
        || {
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                true,
                (0..out.len()).map(|_| 0.0f64),
            )
            .expect("failed to create buffer c")
        },
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

    let finish = vulkano::sync::now(device);
    let finish = finish
        .then_execute(exc.queue.clone(), command_buffer)
        .unwrap();
    let finish = finish
        .then_signal_fence_and_flush()
        .expect("failed to flush");

    // process the remaning columns on the cpu while we await the gpu execution

    // note that the last block is calculated twice, the auther deems this ok for now
    // last columns
    let remaining_n = n % LOCAL_SIZE_X;
    let offset_n = (n - remaining_n) as usize;
    for i in 0..remaining_n {
        let i = i as usize + offset_n;
        for j in 0..p {
            let j = j as usize;
            let mut val = 0.0;
            for k in 0..m {
                let k = k as usize;
                let val0 = values0[i * m as usize + k];
                let val1 = values1[k * p as usize + j];
                val += val0 * val1
            }
            out[i * p as usize + j] = val;
        }
    }
    // last rows
    let remaining_p = p % LOCAL_SIZE_Y;
    let offset_p = (p - remaining_p) as usize;
    for i in 0..n {
        let i = i as usize;
        for j in 0..remaining_p {
            let j = j as usize + offset_p;
            let mut val = 0.0;
            for k in 0..m {
                let k = k as usize;
                let val0 = values0[i * m as usize + k];
                let val1 = values1[k * p as usize + j];
                val += val0 * val1
            }
            out[i * p as usize + j] = val;
        }
    }

    finish.wait(None).expect("compute shader execution failed");

    // gpu finished processing, copy the result
    let content = c_buffer.read().unwrap();
    out.par_iter_mut()
        .enumerate()
        // only set the out values we haven't touched in the cpu-computations
        .take(offset_n * p as usize)
        .for_each(|(i, b)| {
            // TODO pls use chunks
            *b += content[i];
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
    fn test_identity_returns_the_og_matrix() {
        const N: u32 = LOCAL_SIZE_X + 11; // force combined GPU+CPU computation

        let mut rng = rand::thread_rng();
        let a =
            NdArray::new_with_values([N, N], (0..N * N).map(|_| rng.gen_range(0., 10.)).collect())
                .unwrap();

        let mut b =
            NdArray::new_with_values(&[N, N][..], (0..N * N).map(|_| 0.0).collect()).unwrap();
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    *b.get_mut(&[i, j]).unwrap() = 1.0
                }
            }
        }

        let mut c_gpu = NdArray::new([N, N]);
        matmul_f64_impl([N, N, N], a.as_slice(), b.as_slice(), c_gpu.as_mut_slice()).unwrap();

        println!("{}", c_gpu);

        for (c, a) in c_gpu
            .as_slice()
            .iter()
            .zip(a.as_slice().iter())
            .map(|(c, a)| (*c, *a))
        {
            assert!((a - c).abs() < 0.000001)
        }
    }

    #[test]
    fn test_correctness() {
        const N: u32 = LOCAL_SIZE_X + 11; // force combined GPU+CPU computation
        const M: u32 = 70;
        const P: u32 = 30;

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

        for (i, (a, b)) in c_gpu
            .as_slice()
            .iter()
            .zip(c_cpu.as_slice().iter())
            .enumerate()
        {
            assert!((a - b).abs() < 0.000001, "{}: {} != {}", i, a, b)
        }
    }
}
