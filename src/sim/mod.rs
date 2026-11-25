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

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug)]
pub struct Field {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Field {
    pub fn get(&self, p: Point) -> Result<f32, SimError> {
        Ok(self.data[self.offset_from_point(p)?])
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().map(|&v| v as f64).sum()
    }

    pub fn offset_from_point(&self, p: Point) -> Result<usize, SimError> {
        if p.x >= self.width {
            return Err(SimError::InvalidArgument);
        }

        if p.y >= self.height {
            return Err(SimError::InvalidArgument);
        }

        Ok(p.y * self.width + p.x)
    }
}

pub struct Simulation {
    handle: NonNull<ffi::SimHandle>,
    width: usize,
    height: usize,
}

impl Simulation {
    pub fn new(device: i32, width: usize, height: usize, r: f32) -> Result<Self, SimError> {
        let c_width = i32::try_from(width).map_err(|_| SimError::InvalidArgument)?;
        let c_height = i32::try_from(height).map_err(|_| SimError::InvalidArgument)?;
        let settings = ffi::SimSettings {
            device,
            width: c_width,
            height: c_height,
            r,
        };

        let mut raw: *mut ffi::SimHandle = std::ptr::null_mut();

        // SAFETY
        // "settings" is valid, aligned "SimSettings" that lives for the whole call
        // "raw" is a valid destination for writing a *mut SimHandle
        // C doesn't keep either pointer after the call ends
        let status = unsafe { ffi::sim_init(&settings, &mut raw) };
        check(status)?;

        let handle = NonNull::new(raw).ok_or(SimError::Creation)?;
        Ok(Self {
            handle,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn len(&self) -> usize {
        self.width * self.height
    }

    pub fn fill_rect(
        &mut self,
        x: usize,
        y: usize,
        size_x: usize,
        size_y: usize,
        value: f32,
    ) -> Result<(), SimError> {
        let x = i32::try_from(x).map_err(|_| SimError::InvalidArgument)?;
        let y = i32::try_from(y).map_err(|_| SimError::InvalidArgument)?;
        let size_x = i32::try_from(size_x).map_err(|_| SimError::InvalidArgument)?;
        let size_y = i32::try_from(size_y).map_err(|_| SimError::InvalidArgument)?;

        // SAFETY
        // "handle" came from "sim_init" and hasn't been destroyed.
        // &mut self guarantees no simultaneous access during the call.
        //
        let status =
            unsafe { ffi::sim_fill_rect(self.handle.as_ptr(), x, y, size_x, size_y, value) };
        check(status)
    }

    pub fn download(&self) -> Result<Field, SimError> {
        let mut v = vec![0.0; self.len()];
        self.download_into(&mut v)?;
        Ok(Field {
            width: self.width,
            height: self.height,
            data: v,
        })
    }

    pub fn download_into(&self, dst: &mut [f32]) -> Result<(), SimError> {
        // SAFETY
        // "handle" came from "sim_init" and hasn't been destroyed.
        // &Simulation isn't Sync, so cannot be shared across threads
        // "dst" is a valid location for writing "dst.len()" floats into, coming from a live &mut slice
        // of that length.
        // C writes nothing past "dst.len()"
        let status =
            unsafe { ffi::sim_download(self.handle.as_ptr(), dst.as_mut_ptr(), dst.len()) };
        check(status)
    }

    pub fn step(&mut self, count: i32) -> Result<(), SimError> {
        // SAFETY
        // "handle" came from "sim_init" and hasn't been destroyed.
        // &mut self guarantees no simultaneous access during the call.
        let status = unsafe { ffi::sim_step(self.handle.as_ptr(), count) };
        check(status)
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
