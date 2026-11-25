pub type SimResult = i32;

pub const SIM_RESULT_OKAY: SimResult = 0;
pub const SIM_RESULT_INVALID_HANDLE: SimResult = 1;
pub const SIM_RESULT_INVALID_ARGUMENT: SimResult = 2;
pub const SIM_RESULT_CUDA_ERROR: SimResult = 3;
pub const SIM_RESULT_CREATION_ERROR: SimResult = 4;
pub const SIM_RESULT_INVALID_DEVICE: SimResult = 5;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SimSettings {
    pub device: i32,
    pub width: i32,
    pub height: i32,
    pub r: f32,
}

#[repr(C)]
pub struct SimHandle {
    _data: (),
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

unsafe extern "C" {
    pub fn sim_init(settings: *const SimSettings, out_handle: *mut *mut SimHandle) -> SimResult;
    pub fn sim_fill_rect(
        handle: *mut SimHandle,
        x: i32,
        y: i32,
        size_x: i32,
        size_y: i32,
        value: f32,
    ) -> SimResult;
    pub fn sim_step(handle: *mut SimHandle, count: i32) -> SimResult;
    pub fn sim_download(handle: *const SimHandle, dst: *mut f32, dst_count: usize) -> SimResult;
    pub fn sim_destroy(handle: *mut SimHandle);
}

const _: () = assert!(core::mem::size_of::<SimSettings>() == 16);
