use group_traits::*;
use naga_oil::compose::Composer;
use prelude::AccessType;
use std::borrow::Cow;

use std::fmt::Debug;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};
use std::sync::{Arc, Once};
use std::time::Duration;

use wgpu::naga;
use wgpu::naga::valid::Capabilities;

use notify_debouncer_mini::{new_debouncer, notify::*, Debouncer};

use linkme::distributed_slice;

pub use bytemuck;

#[distributed_slice]
pub static SHADERS: [Shader] = [..];

#[macro_export]
macro_rules! shader_file {
    ($v:vis $asdf:ident $file:literal) => {
        #[$crate::m::linkme::distributed_slice($crate::SHADERS)]
        #[linkme(crate = $crate::m::linkme)]
        $v static $asdf: $crate::Shader = $crate::Shader::from_path(file!(), $file, "", None);
    };
    ($v:vis $asdf:ident $file:literal $comp:expr) => {
        #[$crate::m::linkme::distributed_slice($crate::SHADERS)]
        #[linkme(crate = $crate::m::linkme)]
        $v static $asdf: $crate::Shader = $crate::Shader::from_path(file!(), $file, "", Some($comp));
    };
}

pub mod prelude {
    pub use crate::access::*;
    pub use crate::group_traits::*;
}

pub mod access {
    pub enum Read {}
    pub enum Write {}
    pub enum ReadWrite {}
    pub trait AccessType {
        fn storage_texture_access() -> wgpu::StorageTextureAccess;
        fn storage_buffer_type() -> wgpu::BufferBindingType;
    }

    impl AccessType for Read {
        fn storage_texture_access() -> wgpu::StorageTextureAccess {
            wgpu::StorageTextureAccess::ReadOnly
        }
        fn storage_buffer_type() -> wgpu::BufferBindingType {
            wgpu::BufferBindingType::Storage { read_only: true }
        }
    }
    impl AccessType for Write {
        fn storage_texture_access() -> wgpu::StorageTextureAccess {
            wgpu::StorageTextureAccess::WriteOnly
        }
        fn storage_buffer_type() -> wgpu::BufferBindingType {
            wgpu::BufferBindingType::Storage { read_only: false }
        }
    }
    impl AccessType for ReadWrite {
        fn storage_texture_access() -> wgpu::StorageTextureAccess {
            wgpu::StorageTextureAccess::ReadWrite
        }
        fn storage_buffer_type() -> wgpu::BufferBindingType {
            wgpu::BufferBindingType::Storage { read_only: false }
        }
    }
}

pub mod group_traits {
    use std::{any::TypeId, collections::BTreeMap, sync::Arc};

    fn with_layout<B, A, F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U
    where
        B: BindLayoutFor<A>,
    {
        B::with_layout_impl(f)
    }

    pub trait BindLayoutEntry {}
    pub trait BindLayoutEntryFor<A>: BindLayoutEntry {
        fn layout_entry(idx: u32) -> wgpu::BindGroupLayoutEntry;
    }

    impl<T: BindLayoutEntry> BindLayoutEntry for &T {}
    impl<A, T: BindLayoutEntryFor<A>> BindLayoutEntryFor<A> for &T {
        fn layout_entry(idx: u32) -> wgpu::BindGroupLayoutEntry {
            T::layout_entry(idx)
        }
    }

    pub trait BindLayout {}
    pub trait BindLayoutFor<T>: BindLayout {
        fn with_layout_impl<F: for<'a> FnOnce(&'a wgpu::BindGroupLayoutDescriptor) -> U, U>(
            f: F,
        ) -> U;
    }

    pub trait BindEntryFor {
        fn with_resource<F: for<'a> FnOnce(wgpu::BindingResource<'a>) -> U, U>(&self, f: F) -> U;
    }

    impl BindEntryFor for wgpu::Texture {
        fn with_resource<F: for<'a> FnOnce(wgpu::BindingResource<'a>) -> U, U>(&self, f: F) -> U {
            f(wgpu::BindingResource::TextureView(
                &self.create_view(&Default::default()),
            ))
        }
    }
    impl<T: BindEntryFor> BindEntryFor for &T {
        fn with_resource<F: for<'a> FnOnce(wgpu::BindingResource<'a>) -> U, U>(&self, f: F) -> U {
            T::with_resource(self, f)
        }
    }

    pub trait StaticBindEntryFor<A>: BindEntryFor + BindLayoutEntryFor<A> {}
    impl<A, T> StaticBindEntryFor<A> for T where T: BindEntryFor + BindLayoutEntryFor<A> {}

    static LAYOUT_CACHE: parking_lot::RwLock<
        BTreeMap<(TypeId, TypeId), Arc<wgpu::BindGroupLayout>>,
    > = parking_lot::RwLock::new(BTreeMap::new());

    fn type_id<T>() -> TypeId {
        std::any::Any::type_id(&|_: T| ())
    }

    pub trait BindLayoutExt: BindLayout + Sized {
        fn with_layout<A: 'static, F: for<'a> FnOnce(&'a wgpu::BindGroupLayout) -> U, U>(
            device: &wgpu::Device,
            f: F,
        ) -> U
        where
            Self: BindLayoutFor<A>,
        {
            let layout = {
                let cache = LAYOUT_CACHE.upgradable_read();
                let typeid = type_id::<Self>();
                let access_typeid = type_id::<A>();
                if let Some(layout) = cache.get(&(typeid, access_typeid)) {
                    layout.clone()
                } else {
                    let mut cache = parking_lot::RwLockUpgradableReadGuard::upgrade(cache);
                    let layout = Arc::new(with_layout::<Self, A, _, _>(|layout| {
                        device.create_bind_group_layout(layout)
                    }));
                    cache.insert((typeid, access_typeid), layout.clone());
                    layout
                }
            };
            f(&layout)
        }
    }
    impl<B: BindLayout> BindLayoutExt for B {}

    pub trait Bindable: BindLayout {}
    pub trait BindableFor<T>: Bindable + BindLayoutFor<T> {
        fn group_impl(&self) -> &wgpu::BindGroup;
    }
    pub trait BindableExt: Bindable {
        fn group<A>(&self) -> &wgpu::BindGroup
        where
            Self: BindableFor<A>,
        {
            self.group_impl()
        }
    }
    impl<T: Bindable> BindableExt for T {}

    pub struct BoundTuple<T, A> {
        tuple: T,
        group: wgpu::BindGroup,
        _access: std::marker::PhantomData<A>,
    }

    pub trait BindFor<A> {
        type Bound: Bindable + BindableFor<A>;
        fn bind(self, device: &wgpu::Device) -> Self::Bound;
    }

    impl<T: BindLayout, A> BindLayout for BoundTuple<T, A> {}
    impl<T: BindLayoutFor<A>, A> BindLayoutFor<A> for BoundTuple<T, A> {
        fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
            T::with_layout_impl(f)
        }
    }

    macro_rules! bind_tuple {
        ($([$t_idx:tt] $elem:ident),+) => {
            impl<$($elem : BindLayoutEntry),+> BindLayout for ($($elem,)+) {}

            impl<A, $($elem : BindLayoutEntry + BindLayoutEntryFor<A>),+> BindLayoutFor<A> for ($($elem,)+) {
                fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
                    f(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &bind_tuple!(+layout_entries 0, [] $(,$elem)+ )
                        ,
                    })
                }
            }
            impl<A, $($elem : BindLayoutEntry),+> Bindable for BoundTuple<($($elem,)+), A> {}
            impl<A, $($elem : BindLayoutEntryFor<A>),+> BindableFor<A> for BoundTuple<($($elem,)+), A> {
                fn group_impl(&self) -> &wgpu::BindGroup {
                    &self.group
                }
            }
            #[allow(non_snake_case)]
            impl<A: 'static, $($elem : StaticBindEntryFor<A>),+> BindFor<A> for ($($elem,)+) {
                type Bound = BoundTuple<($($elem,)+), A>;
                fn bind(self, device: &wgpu::Device) -> Self::Bound {

                    let group = Self::with_layout::<A, _, _>(device, |layout|{
                        bind_tuple!(+group self device layout 0, [] $(,[$t_idx] $elem)+)
                    });
                    BoundTuple {
                        tuple: self,
                        group,
                        _access: Default::default(),
                    }
                }

            }
        };
        (+layout_entries $next_idx:expr, [$($entry:ident ; $idx:expr,)*] $(,)?) => {
            [$($entry::layout_entry($idx),)*]
        };
        (+layout_entries $next_idx:expr, [$($entry:ident ; $idx:expr,)*], $first:ident $(, $rest:ident)* $(,)? ) => {
            bind_tuple!(+layout_entries $next_idx + 1, [$($entry ; $idx,)* $first ; $next_idx,], $($rest),*)

        };
        (+group $self:ident $device:ident $layout:ident $next_idx:expr, [$([$t_idx:tt] $entry:ident ; $idx:expr,)*] $(,)?) => {
            $device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: $layout,
                entries: &[
                     $(wgpu::BindGroupEntry {
                        binding: $idx,
                        resource: $entry
                    },)*
                ]
            })
        };
        (+tuple_access $self:ident [$x:tt]) => {
            ($self.$x)
        };
        (+group $self:ident $device:ident $layout:ident $next_idx:expr, [$([$t_idx:tt] $entry:ident ; $idx:expr,)*], [$t_first:tt] $first:ident $(, [$rest_idx:tt] $rest:ident)* $(,)? ) => {
            bind_tuple!(+tuple_access $self [$t_first]) .with_resource(|$first| {
                bind_tuple!(+group $self $device $layout $next_idx + 1, [$([$t_idx] $entry ; $idx,)* [$t_first] $first ; $next_idx,], $([$rest_idx] $rest),*)
            })
        };
        (+tuple [$t_idx:tt] $first:ident) => {
            bind_tuple!([$t_idx] $first);
        };
        (+tuple [$t_idx:tt] $first:ident, $([$rest_idx:tt] $rest:ident),+) => {
            bind_tuple!([$t_idx] $first, $([$rest_idx] $rest),+);
            bind_tuple!(+tuple $([$rest_idx] $rest),+);
        }
    }
    // bind_tuple!(+tuple
    //     [0] B00
    // );
    bind_tuple!(+tuple
        [15] B15,
        [14] B14,
        [13] B13,
        [12] B12,
        [11] B11,
        [10] B10,
        [9] B09,
        [8] B08,
        [7] B07,
        [6] B06,
        [5] B05,
        [4] B04,
        [3] B03,
        [2] B02,
        [1] B01,
        [0] B00
    );
}

#[doc(hidden)]
pub mod m {
    pub use linkme;
    pub use log;

    pub use naga_oil::compose::{Composer, ShaderLanguage};
    #[cfg(debug_assertions)]
    #[macro_export]
    macro_rules! shader_bytes_inc {
        ($file:literal) => {{
            None
        }};
    }
    #[cfg(not(debug_assertions))]
    #[macro_export]
    macro_rules! shader_bytes_inc {
        ($file:literal) => {{
            Some(include_str!($file))
        }};
    }
    #[macro_export]
    macro_rules! compose_modifier {
        ($name:ident [$($path:literal $($kind:ident)?),+ $(,)?]) => {
            pub struct $name;
            impl $name {
                const MODULES: &'static [$crate::DepModule] = &[
                    $(
                        $crate::compose_modifier!(+_mod $path $($kind)?),
                    )+
                ];
            }
            impl $crate::ComposerModifier for $name {
                fn modify(&self, composer: &mut $crate::m::Composer) {
                    for module in Self::MODULES {
                        module.add_to(composer);
                    }
                }
                fn dependencies(&self) -> Option<Vec<$crate::DepModule>> {
                    Some(Self::MODULES.to_vec())
                }
            }
        };
        (+_mod $path:literal $($kind:ident)?) => {
                $crate::DepModule::new(
                    $path,
                    file!(),
                    $crate::shader_bytes_inc!($path),
                    $crate::compose_modifier!(+_lang $($kind)?),
                )
        };
        (+_lang) => {
            $crate::m::ShaderLanguage::Wgsl
        };
        (+_lang wgsl) => {
            $crate::m::ShaderLanguage::Wgsl
        };
        (+_lang glsl) => {
            $crate::m::ShaderLanguage::Glsl
       };
    }
    #[macro_export]
    macro_rules! compute_pipelines {
        ($p:ident {$($entry:ident),* $(,)?}) => {
            #[derive(Debug)]
            struct $p {
                $($entry: wgpu::ComputePipeline,)*
            }
            impl $crate::Loadable<&wgpu::PipelineLayout> for $p {
                fn load(
                    device: &wgpu::Device,
                    shader: wgpu::ShaderModule,
                    pipeline_layout: &wgpu::PipelineLayout,
                ) -> Self {
                    Self {
                        $($entry: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(concat!(stringify!($entry), " ", "(", module_path!(), "::", stringify!($p), " ", file!(), ":", line!(), ")")),
                            layout: Some(&pipeline_layout),
                            module: &shader,
                            entry_point: stringify!($entry),
                        }),
                        )*

                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! render_pipeline {
        ($p:ident {$($pipeline_name:ident : $vertex_entry:ident -> $fragment_entry:ident ),* $(,)?}) => {
            #[derive(Debug)]
            struct $p {
                $($pipeline_name: wgpu::RenderPipeline,)*
            }
            impl $crate::Loadable<&wgpu::PipelineLayout> for $p {
                fn load(
                    device: &wgpu::Device,
                    shader: wgpu::ShaderModule,
                    pipeline_layout: &wgpu::PipelineLayout,
                ) -> Self {
                    Self {
                        $($pipeline_name: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(stringify!($vertex_entry -> $fragment_entry)),
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: stringify!($vertex_entry),
                                buffers: &[],
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: stringify!($fragment_entry),
                                targets: &[]
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                ..Default::default()
                            },
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: wgpu::TextureFormat::Depth32Float,
                                depth_write_enabled: true,
                                depth_compare: wgpu::CompareFunction::Greater,
                                stencil: Default::default(),
                                bias: Default::default(),
                            }),
                            multisample: wgpu::MultisampleState::default(),
                            multiview: None,
                        }),
                        )*

                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! create_pipeline_layout {
        ([$device:ident] layouts: {$($type:ty : $access:ident),* $(,)?} $(push_constant_ranges: $pcr:expr)?  ) => {
            $crate::create_pipeline_layout!(+inner $device [] $($type : $access),* $($pcr)?)
        };
        (+inner $device:ident [$($layout:ident),*] $first_ty:ty : $first_access:ident $(, $type:ty : $access:ident)* $($pcr:expr)?) => {
            <$first_ty>::with_layout::<$first_access, _, _>($device, |layout_ident| {
                $crate::create_pipeline_layout!(+inner $device [$($layout,)* layout_ident] $($type : $access ),* $($pcr)?)
            })
        };
        (+inner $device:ident [$($layout:ident),*] $pcr:expr) => {
            $device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    $($layout,)*
                ],
                push_constant_ranges: $pcr,
            })
        };
        (+inner $device:ident [$($layout:ident),*] ) => {
            $device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    $($layout)*,
                ],
                push_constant_ranges: &[],
            })
        };
    }
}

pub fn composer() -> Composer {
    let composer = Composer::default().with_capabilities(Capabilities::all());
    composer
}

#[doc(hidden)]
pub mod d {
    use std::path::PathBuf;

    use naga_oil::compose::{
        ComposableModuleDefinition, ComposableModuleDescriptor, Composer, ShaderLanguage,
    };

    #[derive(Debug, Clone)]
    pub struct DepModule<'a> {
        pub(crate) shader_path_relative: &'a str,
        pub(crate) shader_path_base: &'a str,
        pub(crate) source: Option<&'a str>,
        pub(crate) language: ShaderLanguage,
    }

    impl<'a> DepModule<'a> {
        pub const fn new(
            shader_path_relative: &'a str,
            shader_path_base: &'a str,
            source: Option<&'a str>,
            language: ShaderLanguage,
        ) -> Self {
            Self {
                shader_path_relative,
                shader_path_base,
                source,
                language,
            }
        }
        fn path(&self) -> PathBuf {
            let shader_path = AsRef::<std::path::Path>::as_ref(self.shader_path_base)
                .parent()
                .unwrap()
                .join(self.shader_path_relative);
            shader_path
        }
        pub fn add_to<'b>(
            &self,
            composer: &'b mut Composer,
        ) -> Option<&'b ComposableModuleDefinition> {
            let path = self.path();
            let live_source = path.exists().then_some(()).and_then(|()| {
                std::fs::read_to_string(&path)
                    .map_err(|e| {
                        log::error!(
                            "Unable to load live shader source for module from {}: {}",
                            path.display(),
                            e
                        );
                    })
                    .ok()
            });
            let source = live_source.as_deref().or(self.source);

            if let Some(source) = source {
                return composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source,
                        file_path: path.to_str().unwrap(),
                        language: self.language,
                        ..Default::default()
                    })
                    .map_err(|e| {
                        log::error!("Unable to compose shader from {}: {}", path.display(), e);
                    })
                    .ok();
            } else {
                log::error!(
                    "Unable to load built in or live source for shader: {}",
                    self.shader_path_relative
                );
                return None;
            }
        }
    }
}
pub use util::GpuData;

pub mod util {
    use bytemuck::Pod;
    use std::{marker::PhantomData, mem};

    use crate::prelude::*;

    pub struct GpuData<T> {
        read_group: wgpu::BindGroup,
        read_write_group: wgpu::BindGroup,
        buffer: wgpu::Buffer,
        _marker: PhantomData<T>,
    }

    impl<T: Pod> GpuData<T> {
        pub fn new(device: &wgpu::Device, data: &T) -> Self {
            use wgpu::util::DeviceExt;
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Control buffer"),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let make_group = |layout: &_| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Control Data"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                    }],
                })
            };

            let read_write_group = Self::with_layout::<ReadWrite, _, _>(device, make_group);
            let read_group = Self::with_layout::<Read, _, _>(device, make_group);
            Self {
                read_group,
                read_write_group,
                buffer,
                _marker: Default::default(),
            }
        }
        pub fn into_inner(self) -> wgpu::Buffer {
            self.buffer
        }
    }

    impl<T> BindLayout for GpuData<T> {}
    impl<A: AccessType, T> BindLayoutFor<A> for GpuData<T> {
        fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
            f(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Reproject Control"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: A::storage_buffer_type(),
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(mem::size_of::<Self>() as u64),
                    },
                    count: None,
                }],
            })
        }
    }
    impl<T> Bindable for GpuData<T> {}
    impl<T> BindableFor<ReadWrite> for GpuData<T> {
        fn group_impl(&self) -> &wgpu::BindGroup {
            &self.read_write_group
        }
    }
    impl<T> BindableFor<Read> for GpuData<T> {
        fn group_impl(&self) -> &wgpu::BindGroup {
            &self.read_group
        }
    }
}

pub type DepModule = d::DepModule<'static>;

pub trait ComposerModifier: Send + Sync {
    fn modify(&self, composer: &mut Composer);
    fn dependencies(&self) -> Option<Vec<DepModule>>;
}

#[derive(Debug)]
pub struct PipelineCache<T> {
    pipeline: parking_lot::RwLock<Option<Arc<T>>>,
    shader: &'static Shader,
}

#[derive(Debug)]
pub struct PipelineCacheWithLayout<T> {
    cache: PipelineCache<T>,
    layout: wgpu::PipelineLayout,
}

impl<T> PipelineCacheWithLayout<T> {
    pub fn load<'a>(&'a self, device: &wgpu::Device) -> impl std::ops::Deref<Target = T> + 'a
    where
        T: Loadable<&'a wgpu::PipelineLayout>,
    {
        self.cache.load_auto(device, &self.layout)
    }
}

pub trait Loadable<R> {
    fn load(device: &wgpu::Device, shader: wgpu::ShaderModule, _: R) -> Self;
}

impl<T> PipelineCache<T> {
    pub fn load_auto<R>(&self, device: &wgpu::Device, r: R) -> impl std::ops::Deref<Target = T>
    where
        T: Loadable<R>,
    {
        let pipeline = self.pipeline.upgradable_read();
        if self.shader.should_reload() || pipeline.is_none() {
            let mut pipeline = parking_lot::RwLockUpgradableReadGuard::upgrade(pipeline);
            let source = self.shader.load();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(self.shader.shader_path),
                source,
            });

            let new_pipeline = Arc::new(T::load(device, shader, r));
            *pipeline = Some(new_pipeline.clone());
            new_pipeline
        } else {
            pipeline.as_ref().unwrap().clone()
        }
    }
}

pub struct Shader {
    source_path: &'static str,
    shader_path: &'static str,
    text: &'static str,
    included: OnceLock<naga::Module>,
    notify_launched: Once,
    live: Mutex<Option<naga::Module>>,
    debouncer: OnceLock<Debouncer<RecommendedWatcher>>,
    flag: AtomicBool,
    comp: Option<&'static dyn ComposerModifier>,
}
impl Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shader")
            .field("source_path", &self.source_path)
            .field("shader_path", &self.shader_path)
            .finish()
    }
}

impl Shader {
    pub fn cache<T>(&'static self) -> PipelineCache<T> {
        PipelineCache {
            pipeline: None.into(),
            shader: self,
        }
    }
    pub fn cache_with_layout<T>(
        &'static self,
        layout: wgpu::PipelineLayout,
    ) -> PipelineCacheWithLayout<T> {
        PipelineCacheWithLayout {
            cache: self.cache(),
            layout,
        }
    }
    pub const fn from_path(
        source_path: &'static str,
        shader_path: &'static str,
        text: &'static str,
        comp: Option<&'static dyn ComposerModifier>,
    ) -> Self {
        Self {
            source_path,
            shader_path,
            text,
            included: OnceLock::new(),
            notify_launched: Once::new(),
            live: Mutex::new(None),
            debouncer: OnceLock::new(),
            flag: AtomicBool::new(false),
            comp,
        }
    }
    pub fn try_load(&'static self) -> Option<wgpu::ShaderSource> {
        self.notify_launched.call_once(|| self.setup_notify());
        let mut live_module = self.live.lock().ok()?;

        if let Some(new_module) = self.load_live_bytes() {
            *live_module = Some(new_module);
        }
        // we tried loading, so don't try again until the file has changed
        self.flag.store(false, Ordering::Release);

        live_module
            .as_ref()
            .map(|live| wgpu::ShaderSource::Naga(Cow::Owned(live.clone())))
    }
    pub fn load_static(&'static self) -> wgpu::ShaderSource {
        let included = self.included.get_or_init(|| {
            let mut composer = composer();
            if let Some(m) = self.comp {
                m.modify(&mut composer);
            }
            let mmod = composer
                .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                    source: self.text,
                    file_path: self.shader_path,
                    shader_type: naga_oil::compose::ShaderType::Wgsl,
                    ..Default::default()
                })
                .map_err(|e| {
                    log::warn!(
                        "Unable to compile bundled source for {}: {}",
                        self.shader_path,
                        e.emit_to_string(&composer)
                    );
                })
                .unwrap();
            //naga::front::wgsl::Frontend::new().parse(self.text).unwrap()
            mmod
        });
        wgpu::ShaderSource::Naga(Cow::Owned(included.clone()))
    }
    pub fn load(&'static self) -> wgpu::ShaderSource {
        log::info!("Loading shader {:?}", self.shader_path);
        match self.try_load() {
            Some(s) => s,
            None => self.load_static(),
        }
    }

    fn setup_notify(&'static self) {
        self.debouncer.get_or_init(|| {
            log::info!("Initializing notifier for {:?}", self.shader_path);
            let flag: &'static AtomicBool = &self.flag;

            let mut debouncer = new_debouncer(Duration::from_millis(200), None, move |events| {
                if let Ok(_events) = events {
                    std::thread::sleep(Duration::from_millis(50));
                    flag.store(true, Ordering::Release);
                }
            })
            .unwrap();
            debouncer
                .watcher()
                .watch(
                    &Path::new(self.source_path)
                        .parent()
                        .unwrap()
                        .join(self.shader_path),
                    RecursiveMode::NonRecursive,
                )
                .expect(&format!(
                    "Unable to watch for updates to shader {} defined in {}",
                    self.shader_path, self.source_path,
                ));
            debouncer
        });
    }

    fn load_live_bytes(&self) -> Option<naga::Module> {
        let path = Path::new(self.source_path);
        let source = if let Ok(bytes) = std::fs::read(path.parent().unwrap().join(self.shader_path))
        {
            String::from_utf8(bytes).ok()?
        } else {
            return None;
        };
        let mut composer = composer();
        if let Some(m) = self.comp {
            m.modify(&mut composer);
        }
        let module = match composer.make_naga_module(naga_oil::compose::NagaModuleDescriptor {
            source: &source,
            file_path: self.shader_path,
            shader_type: naga_oil::compose::ShaderType::Wgsl,
            ..Default::default()
        }) {
            Ok(module) => module,
            Err(e) => {
                log::error!(
                    "Error parsing module {:?} \n\n{}\n ^^^^^^^^^^^^^^^^^^^^^^ end of errors for {0:?} ",
                    self.shader_path,
                    e.emit_to_string(&composer)
                );
                return None;
            }
        };
        match naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        {
            Err(e) => {
                log::error!(
                    "Error validating module {:?} \n\n{}\n ^^^^^^^^^^^^^^^^^^^^^^ end of errors for {0:?}",
                    self.shader_path,
                    e.emit_to_string_with_path(&source, self.shader_path)
                );
                return None;
            }
            _ => (),
        };
        return Some(module);
    }
    pub fn should_reload(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

pub struct TypedTexture<T> {
    texture: wgpu::Texture,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TypedTexture<T> {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

impl<T: TextureFormat> BindEntryFor for TypedTexture<T> {
    fn with_resource<F: for<'a> FnOnce(wgpu::BindingResource<'a>) -> U, U>(&self, f: F) -> U {
        self.texture.with_resource(f)
    }
}
impl<T: TextureFormat> BindLayoutEntry for TypedTexture<T> {}
impl<A: AccessType, T: TextureFormat> BindLayoutEntryFor<A> for TypedTexture<T> {
    fn layout_entry(idx: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: idx,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::StorageTexture {
                access: A::storage_texture_access(),
                format: T::FORMAT,
                view_dimension: T::D::VIEW_DIMENSION,
            },
            count: None,
        }
    }
}

impl BindEntryFor for wgpu::Buffer {
    fn with_resource<F: for<'a> FnOnce(wgpu::BindingResource<'a>) -> U, U>(&self, f: F) -> U {
        f(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self,
            offset: 0,
            size: None,
        }))
    }
}
impl BindLayoutEntry for wgpu::Buffer {}
impl<A: AccessType> BindLayoutEntryFor<A> for wgpu::Buffer {
    fn layout_entry(idx: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: idx,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: A::storage_buffer_type(),
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

pub enum D1 {}
pub enum D2 {}
pub enum D3 {}
pub trait Dimension {
    const DIMENSION: wgpu::TextureDimension;
    const VIEW_DIMENSION: wgpu::TextureViewDimension;
}

impl Dimension for D1 {
    const DIMENSION: wgpu::TextureDimension = wgpu::TextureDimension::D1;
    const VIEW_DIMENSION: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D1;
}
impl Dimension for D2 {
    const DIMENSION: wgpu::TextureDimension = wgpu::TextureDimension::D2;
    const VIEW_DIMENSION: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D2;
}
impl Dimension for D3 {
    const DIMENSION: wgpu::TextureDimension = wgpu::TextureDimension::D3;
    const VIEW_DIMENSION: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D3;
}

pub enum RGBA8Uint<D> {
    D(D),
}
pub enum RGBA8Unorm<D> {
    D(D),
}
pub trait TextureFormat {
    type D: Dimension;
    const FORMAT: wgpu::TextureFormat;
    const VIEW_FORMATS: &'static [wgpu::TextureFormat] = &[];
}

impl<D: Dimension> TextureFormat for RGBA8Uint<D> {
    type D = D;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
}

impl<D: Dimension> TextureFormat for RGBA8Unorm<D> {
    type D = D;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
}

impl TypedTexture<()> {
    pub fn new_1d<T: TextureFormat<D = D1>>(device: &wgpu::Device, width: u32) -> TypedTexture<T> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: D1::DIMENSION,
            format: T::FORMAT,
            usage: wgpu::TextureUsages::all(),
            view_formats: T::VIEW_FORMATS,
        });
        TypedTexture {
            texture,
            _marker: Default::default(),
        }
    }
    pub fn new<T: TextureFormat<D = D2>>(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> TypedTexture<T> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: D2::DIMENSION,
            format: T::FORMAT,
            usage: wgpu::TextureUsages::all(),
            view_formats: T::VIEW_FORMATS,
        });
        TypedTexture {
            texture,
            _marker: Default::default(),
        }
    }
    pub fn new_3d<T: TextureFormat<D = D3>>(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
    ) -> TypedTexture<T> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: D3::DIMENSION,
            format: T::FORMAT,
            usage: wgpu::TextureUsages::all(),
            view_formats: T::VIEW_FORMATS,
        });
        TypedTexture {
            texture,
            _marker: Default::default(),
        }
    }
}

pub struct UploadedBuffer {
    buffer: wgpu::Buffer,
}

impl UploadedBuffer {
    pub fn new<T: bytemuck::Pod + bytemuck::Zeroable>(device: &wgpu::Device, data: &[T]) -> Self {
        let data_bytes: &[u8] = bytemuck::cast_slice(data);
        let mut len = data_bytes.len() as u64;
        if len % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            len += wgpu::COPY_BUFFER_ALIGNMENT - (len % wgpu::COPY_BUFFER_ALIGNMENT);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(std::any::type_name::<T>()),
            usage: wgpu::BufferUsages::STORAGE,
            size: len,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data_bytes);
        Self { buffer }
    }
    pub fn finish(self) -> wgpu::Buffer {
        self.buffer
    }
}
