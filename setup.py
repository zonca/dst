from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

numpy_inc = get_include()

ext_modules = [
Extension("accumulate",   ["accumulate.pyx"],   include_dirs=[numpy_inc], extra_compile_args=["-O3",]),
Extension("signalremove", ["signalremove.pyx"], include_dirs=[numpy_inc], extra_compile_args=["-O3",]),
]

setup(
  name = 'Accumulate',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
