from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "_core",  # Module name (build in current directory)
        ["core.cpp"],  # Source files
        include_dirs=[os.path.join(os.path.dirname(__file__), "eigen")],
        extra_compile_args=["/O2" if os.name == 'nt' else "-O3", "-march=native"],
    ),
]

setup(
    name="glassbox_sr_core",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
