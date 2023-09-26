use wgpu_shader_boilerplate as wgs;
use wgpu_shader_boilerplate::bytemuck::{self, Pod, Zeroable};
use wgpu_shader_boilerplate::prelude::*;

use buffer::WorkingBuffers;

fn main() {
    env_logger::init();

    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(async {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,

                    features: wgpu::Features::PUSH_CONSTANTS,
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    limits: wgpu::Limits {
                        max_push_constant_size: 64,
                        ..Default::default()
                    }
                    .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .unwrap();
        (adapter, device, queue)
    });

    let my_shader = MyShader::new(&device);

    let my_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("My buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    {
        let mut slice = my_buffer.slice(..).get_mapped_range_mut();
        for i in 0..slice.len() {
            slice[i] = (i % 256) as u8;
        }
    }
    my_buffer.unmap();

    let working_data = WorkingBuffers::new(&device, 128);

    let texture = wgs::TypedTexture::new(&device, 256, 256);

    device.poll(wgpu::Maintain::Wait);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("My command encoder"),
        ..Default::default()
    });

    my_shader.run(&device, &mut encoder, &my_buffer, &working_data, &texture);

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
}

#[cfg(dont_actually_compile_this)]
mod example {
    use wgpu_shader_boilerplate as wgs;
    // Define a set of imports (optional)
    // The defined struct is always pub
    wgs::compose_modifier! {
        CommonImports [
            "common.wgsl",
            "util.wgsl" wgsl,
            "tools.glsl" glsl,
        ]
    }

    // Load compute.wgsl, which may import items from CommonImports
    wgs::shader_file!(pub SHADER "shader.wgsl" &CommonImports);
    // Load dummy.wgsl, which imports nothing and is public only within this crate
    wgs::shader_file!(pub(crate) DUMMY "dummy.wgsl");
    // load priv.wgsl
    wgs::shader_file!(PRIV "priv.wgsl");
}
wgs::shader_file!(pub COMPUTE "compute.wgsl");

// Define pipeline with entrypoints precompute_pass1, precompute_pass2,
wgs::compute_pipelines!(Pipelines {
    compute_pass1,
    compute_pass2,
    compute_unused_pass,
});

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct DispatchData {
    flags: u32,
    texture_size: [u32; 2],
}

struct MyShader {
    pipeline: wgs::PipelineCacheWithLayout<Pipelines>,
}
impl MyShader {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipeline = COMPUTE.cache_with_layout(wgs::create_pipeline_layout!(
            [device]
            layouts: {
                (wgpu::Buffer,): Read, // input date
                WorkingBuffers: ReadWrite, //
                (wgs::TypedTexture<wgs::RGBA8Unorm<wgs::D2>>,): Write // output
            }
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..(std::mem::size_of::<DispatchData>() as u32),
            }]

        ));
        Self { pipeline }
    }
    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_buffer: &wgpu::Buffer,
        working_buffers: &WorkingBuffers,
        texture: &wgs::TypedTexture<wgs::RGBA8Unorm<wgs::D2>>,
    ) {
        let texture_size = {
            let sz = texture.texture().size();
            [sz.width, sz.height]
        };
        // this block scope is important
        {
            let Pipelines {
                compute_pass1,
                compute_pass2,
                ..
            } = &*self.pipeline.load(device);

            let in_group = (input_buffer,).bind(device);
            let texture_group = (texture,).bind(device);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("example compute pass"),
                ..Default::default()
            });

            cpass.set_pipeline(compute_pass1);
            cpass.set_push_constants(
                0,
                bytemuck::bytes_of(&DispatchData {
                    flags: 0,
                    texture_size,
                }),
            );
            cpass.set_bind_group(0, in_group.group::<Read>(), &[]);
            cpass.set_bind_group(1, working_buffers.group::<ReadWrite>(), &[]);
            cpass.set_bind_group(2, texture_group.group::<Write>(), &[]);
            cpass.dispatch_workgroups(10, 1, 1);

            cpass.set_pipeline(compute_pass2);
            cpass.set_push_constants(
                0,
                bytemuck::bytes_of(&DispatchData {
                    flags: 1,
                    texture_size,
                }),
            );
            cpass.dispatch_workgroups(10, 1, 1);
        }
    }
}

mod buffer {
    use wgpu_shader_boilerplate::prelude::*;
    pub struct WorkingBuffers {
        pub buf1: wgpu::Buffer,
        pub buf2: wgpu::Buffer,

        read_write_group: wgpu::BindGroup,
        read_group: wgpu::BindGroup,
    }

    impl WorkingBuffers {
        pub fn new(device: &wgpu::Device, count: u32) -> Self {
            let create_buffer = |size| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("working buffers"),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };

            let buf1 = create_buffer(count * 4);
            let buf2 = create_buffer((count * 2) * 4);

            let make_descriptor = |layout: &_| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("working data bind group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                buf1.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(
                                buf2.as_entire_buffer_binding(),
                            ),
                        },
                    ],
                })
            };

            let read_group =
                Self::with_layout::<Read, _, _>(device, |layout| make_descriptor(layout));
            let read_write_group =
                Self::with_layout::<ReadWrite, _, _>(device, |layout| make_descriptor(layout));

            Self {
                buf1,
                buf2,
                read_group,
                read_write_group,
            }
        }
    }

    impl BindLayout for WorkingBuffers {}
    impl<T: AccessType> BindLayoutFor<T> for WorkingBuffers {
        fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
            let stages = wgpu::ShaderStages::COMPUTE;
            let entry = |idx| wgpu::BindGroupLayoutEntry {
                binding: idx,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: T::storage_buffer_type(),
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            f(&wgpu::BindGroupLayoutDescriptor {
                label: Some("splat working buffers"),
                entries: &[entry(0), entry(1)],
            })
        }
    }
    impl Bindable for WorkingBuffers {}
    impl BindableFor<Read> for WorkingBuffers {
        fn group_impl(&self) -> &wgpu::BindGroup {
            &self.read_group
        }
    }

    impl BindableFor<ReadWrite> for WorkingBuffers {
        fn group_impl(&self) -> &wgpu::BindGroup {
            &self.read_write_group
        }
    }
}
