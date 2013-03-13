from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("accumulate", ["accumulate.pyx"], extra_compile_args=["-O3",]),
               Extension("signalremove", ["signalremove.pyx"], extra_compile_args=["-O3",]),
]

setup(
  name = 'Accumulate',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
