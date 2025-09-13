
import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

try:
    torch_lib_dir = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
    if not os.path.exists(torch_lib_dir):
        torch_lib_dir = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "..", "lib")
except ImportError:
    torch_lib_dir = None
    print("Warning: PyTorch not found. RPATH will not be set.")

common_cxx_args = ["-O3", "-std=c++17"]
common_nvcc_args = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
]
common_nvcc_args.extend([
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_90,code=sm_90',
])

# 动态添加RPATH链接参数
link_args = []
if torch_lib_dir and os.path.exists(torch_lib_dir):
    link_args.append(f'-Wl,-rpath,{torch_lib_dir}')

# 模块定义
sources = {
    "selective_scan_cuda_core": [
        "csrc/selective_scan/cus/selective_scan.cpp",
        "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
        "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
    ],
    "selective_scan_cuda_ndstate": [
        "csrc/selective_scan/cusndstate/selective_scan_ndstate.cpp",
        "csrc/selective_scan/cusndstate/selective_scan_core_fwd.cu",
        "csrc/selective_scan/cusndstate/selective_scan_core_bwd.cu",
    ],
    "selective_scan_cuda_oflex": [
        "csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
        "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
        "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu",
    ],
}

ext_modules = []
for name, source_files in sources.items():
    ext_modules.append(
        CUDAExtension(
            name=name,
            sources=source_files,
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
            extra_compile_args={
                "cxx": common_cxx_args,
                "nvcc": common_nvcc_args,
            },
            extra_link_args=link_args
        )
    )

setup(
    name="selective_scan",
    version="0.0.3",
    packages=find_packages(),
    author="MzeroMiko & Your Name",
    description="Selective Scan with robust, reproducible build",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
    ],
)