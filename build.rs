use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn cuda_sources(dir: &Path) -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = fs::read_dir(dir)
        .expect("Failed to read the cuda/ directory")
        .map(|entry| entry.expect("failed to read directory entry").path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();
    sources
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda-13.3".to_string());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let sources = cuda_sources(Path::new("cuda"));

    let status = Command::new(format!("{cuda_path}/bin/nvcc"))
        .args(["--lib", "--std=c++17", "-arch=sm_120"])
        .args(["-Xcompiler", "-Wall,-Wextra"])
        .args(&sources)
        .arg("-o")
        .arg(out_dir.join("libsim.a"))
        .status()
        .expect("failed to start nvcc");
    assert!(status.success(), "nvcc failed");

    // Check that no C++ has snuck into the public header
    let status = Command::new("cc")
        .args(["-x", "c", "-std=c11", "-fsyntax-only"])
        .args(["-Wall", "-Wextra", "-pedantic", "-Werror"])
        .arg("-Wno-pragma-once-outside-header")
        .arg("cuda/simulation.h")
        .status()
        .expect("failed to start cc");
    assert!(status.success(), "cuda/simulation.h is not valid C");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={cuda_path}/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=static=sim");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
