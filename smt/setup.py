from setuptools import setup, Extension
import numpy as np

setup(
    ext_modules=[Extension("mobiusmodule", ["mobius.c"])],
    include_dirs=[np.get_include()]  # This line includes NumPy headers
)