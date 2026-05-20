use std::process::Command;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/snn_kernel.cu");

    // We only compile and link the CUDA code if the "cuda" feature is enabled
    #[cfg(feature = "cuda")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let dest_path = Path::new(&out_dir).join("snn_kernel.ptx");

        println!("cargo:warning=Compiling CUDA SNN kernel via nvcc to PTX: {:?}", dest_path);

        let status = Command::new("nvcc")
            .args(&[
                "-ptx",
                "-O3",
                "-arch=sm_52", // Safe baseline architecture supporting most NVIDIA GPUs
                "src/snn_kernel.cu",
                "-o",
                dest_path.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-cfg=cuda_ptx_compiled");
                println!("cargo:warning=CUDA kernel compiled successfully.");
            }
            Ok(s) => {
                panic!("nvcc compilation failed with exit status: {}", s);
            }
            Err(e) => {
                panic!("Failed to execute nvcc. Ensure NVIDIA CUDA Toolkit is installed and nvcc is on PATH. Error: {}", e);
            }
        }
    }
}
