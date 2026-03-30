from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='roiaware_pool3d',
    ext_modules=[
        CUDAExtension(
            name='roiaware_pool3d_cuda',
            sources=[
                'roiaware_pool3d.cpp',
                'roiaware_pool3d_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)