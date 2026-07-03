from setuptools import setup
from Cython.Build import cythonize

setup(
    name="ptr_extractor",
    ext_modules=cythonize("ptr_extractor.pyx"),
)