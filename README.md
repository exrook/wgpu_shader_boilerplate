
This crate tries to alleviate some of the verbosity of defining pipelines and bind groups in wgpu.

It will also automatically live reload your shaders.

Name subject to change

See also the [example](examples/example.rs)

```rust
use wgpu_shader_boilerplate as wsb;

```
Define a set of imports (optional)
The defined struct is always pub
```rust
wgsb::compose_modifier! {
    CommonImports [
        "common.wgsl",
        "util.wgsl" wgsl,
        "tools.glsl" glsl,
    ]
}

```
Load precompute.wgsl, which may import items from CommonImports
```rust
wsb::shader_file!(pub PRECOMPUTE "precompute.wgsl" &CommonImports);
```
Load dummy.wgsl, which imports nothing and is public only within this crate
```rust
wsb::shader_file!(pub(crate) DUMMY "dummy.wgsl");
```
Load priv.wgsl
```rust
wsb::shader_file!(PRIV "priv.wgsl");
```

Define pipeline with entrypoints precompute_pass1, precompute_pass2,
```rust
wsb::compute_pipelines!(Pipelines {
    precompute_pass1,
    precompute_pass2,
});
```

```rust
struct MyShader {
    pipeline: wsb::PipelineCacheWithLayout<Pipelines>,
}
impl MyShader {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipeline = PRECOMPUTE.cache_with_layout(wsb::create_pipeline_layout!(
            [device]
            layouts: {
                (wgpu::Buffer,): Read, // input date
                BoundControlData: Read, // 
                (shader::TypedTexture<shader::RGBA8Unorm<shader::D2>>,): ReadWrite // output
            }
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..(mem::size_of::<DispatchData>() as u32),
            }]

        ));
        Self { pipeline }
    }
}
```
