from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            name="robust_multignn_ports_cpp",
            sources=["robust_multignn/ports_cpp.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
