mod ffi;

use std::fmt;
use std::ptr::NonNull;

#[derive(Debug)]
pub enum SimError {
    InvalidHandle,
    InvalidArgument,
    Cuda,
    Creation,
    InvalidDevice,
    Unknown(i32),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(code) => write!(f, "unknown sim error {code}"),
            other => fmt::Debug::fmt(other, f),
        }
    }
}

impl std::error::Error for SimError {}

fn check(code: ffi::SimResult) -> Result<(), SimError> {
    match code {
        ffi::SIM_RESULT_INVALID_HANDLE => Err(SimError::InvalidHandle),
        ffi::SIM_RESULT_INVALID_ARGUMENT => Err(SimError::InvalidArgument),
        ffi::SIM_RESULT_CUDA_ERROR => Err(SimError::Cuda),
        ffi::SIM_RESULT_CREATION_ERROR => Err(SimError::Creation),
        ffi::SIM_RESULT_INVALID_DEVICE => Err(SimError::InvalidDevice),
        ffi::SIM_RESULT_OKAY => Ok(()),
        _ => Err(SimError::Unknown(code)),
    }
}

pub struct Simulation {
    handle: NonNull<ffi::SimHandle>,
}

impl Simulation {
    pub fn new(device: i32, width: i32, height: i32, r: f32) -> Result<Self, SimError> {
        let settings = ffi::SimSettings {
            device,
            width,
            height,
            r,
        };

        let mut raw: *mut ffi::SimHandle = std::ptr::null_mut();

        // SAFETY
        // "settings" is valid, aligned "SimSettings" that lives for the whole call
        // "raw" is a valid destination for writing a *mut SimHandle
        // C doesn't keep either pointer after the call ends
        let code = unsafe { ffi::sim_init(&settings, &mut raw) };
        check(code)?;

        let handle = NonNull::new(raw).ok_or(SimError::Creation)?;
        Ok(Self { handle })
    }

    pub fn fill_rect(
        &mut self,
        x: i32,
        y: i32,
        size_x: i32,
        size_y: i32,
        value: f32,
    ) -> Result<(), SimError> {
        // SAFETY
        // "handle" came from "sim_init" and hasn't been destroyed.
        // &mut self guarantees no simultaneous access during the call.
        let result =
            unsafe { ffi::sim_fill_rect(self.handle.as_ptr(), x, y, size_x, size_y, value) };
        check(result)
    }

    pub fn step(&mut self, count: i32) -> Result<(), SimError> {
        // SAFETY
        // "handle" came from "sim_init" and hasn't been destroyed.
        // &mut self guarantees no simultaneous access during the call.
        let result = unsafe { ffi::sim_step(self.handle.as_ptr(), count) };
        check(result)
    }
}

impl Drop for Simulation {
    fn drop(&mut self) {
        // SAFETY
        // "handle" is valid and owned by this Simulation.
        // No double-free because drop() is only called once.
        unsafe {
            ffi::sim_destroy(self.handle.as_ptr());
        }
    }
}
