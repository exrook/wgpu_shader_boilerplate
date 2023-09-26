@group(0) @binding(0)
var<storage, read> input_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read_write> working_buffer1: array<u32>;
@group(1) @binding(1)
var<storage, read_write> working_buffer2: array<u32>;

@group(2) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1,1,1)
fn compute_pass1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    working_buffer1[id.x] = input_buffer[id.x];
    working_buffer2[num_workgroups.x - id.x] = input_buffer[id.x];
}
@compute @workgroup_size(1,1,1)
fn compute_pass2(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let size = textureDimensions(output_texture);
    let x = id.x % size.x;
    let y = id.x / size.x;
    textureStore(output_texture, vec2(x,y), vec4(f32(working_buffer2[id.x] + working_buffer2[id.x]) / 65535.0, 0.0, 0.0, 1.0));
}
@compute @workgroup_size(1,1,1)
fn compute_unused_pass() {

}
