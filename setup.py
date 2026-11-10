"""Install ROSA and build its CUDA extension.

    pip install -e . --no-build-isolation

``--no-build-isolation`` lets the build use the installed PyTorch.  Set
``TORCH_CUDA_ARCH_LIST`` (e.g. "8.6;8.9;12.0") when no GPU is visible.
"""

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

OPS = "projects/mmdet3d_plugin/ops"

setup(
    name="rosa",
    version="1.0.0",
    description=(
        "Ray-aware Observation Selection and Aggregation for "
        "surround-view fisheye 3D detection"
    ),
    python_requires=">=3.10",
    packages=find_packages(include=["rosa*", "projects*"]),
    ext_modules=[
        CUDAExtension(
            name="projects.mmdet3d_plugin.ops.deformable_aggregation_ext",
            sources=[
                f"{OPS}/src/deformable_aggregation.cpp",
                f"{OPS}/src/deformable_aggregation_cuda.cu",
            ],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
