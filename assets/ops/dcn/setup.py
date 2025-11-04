from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(__file__)
src_dir = os.path.join(this_dir, 'src')

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension(
            'deform_conv_cuda',
            [
                os.path.join(src_dir, 'deform_conv_cuda.cpp'),
                os.path.join(src_dir, 'deform_conv_cuda_kernel.cu'),
            ],
        ),
        CUDAExtension(
            'deform_pool_cuda',
            [
                os.path.join(src_dir, 'deform_pool_cuda.cpp'),
                os.path.join(src_dir, 'deform_pool_cuda_kernel.cu'),
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
