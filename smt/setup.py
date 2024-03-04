from setuptools import setup, Extension
import numpy as np

extension_module = Extension(
    "mobiusmodule",
    ["mobius.c"],
    extra_compile_args=["-O3", "-std=c99"]  # Add your optimization flags here
)

setup(
    name="mobiusmodule",
    ext_modules=[extension_module],
    include_dirs=[np.get_include()]  # This line includes NumPy headers
)