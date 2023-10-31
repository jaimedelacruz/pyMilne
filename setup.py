from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

import platform as plt
import sys
import pathlib

os.system('rm pyMilne.*.so pyMilne.*.cpp')
p = pathlib.Path(sys.executable)
root_dir = str(pathlib.Path(*p.parts[0:-2]))


if(plt.system() == 'Darwin'):
    root_dir = '/opt/local/' # using this one if macports are installed
    CC = 'clang'
    CXX= 'clang++'
    link_opts = ["-stdlib=libc++","-bundle","-undefined","dynamic_lookup", "-fopenmp","-lgomp"]
else:
    #root_dir = '/usr/'
    CC = 'gcc'
    CXX= 'g++'
    link_opts = ["-shared", "-fopenmp"]

os.environ["CC"] = CC
os.environ["CXX"] = CXX


# Optimization flags. With Macs M-processor remove the -march=native!

comp_flags=['-Ofast', '-flto','-g0','-fstrict-aliasing','-march=native','-mtune=native','-std=c++20','-fPIC','-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API",'-mprefer-vector-width=256', '-DNDEBUG', '-pedantic', '-Wall']


# Optimization flags for development

#comp_flags=['-Og', '-g3','-fstrict-aliasing','-march=native','-mtune=native','-std=c++20','-fPIC','-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API", '-pedantic', '-Wall']


extension = Extension("pyMilne",
                      sources=["pyMilne.pyx", "src/wrapper_tools_spatially_coupled.cpp", "src/lm_sc.cpp", \
                               "src/spatially_coupled_helper.cpp"], 
                      include_dirs=["./",numpy.get_include(), './eigen3', root_dir+"/include/"],
                      language="c++",
                      extra_compile_args=comp_flags,
                      extra_link_args=comp_flags+link_opts,
                      library_dirs=['./',"/usr/lib/"],
                      libraries=['fftw3'])
#                      undef_macros = [ "NDEBUG" ])

extension.cython_directives = {'language_level': "3"}

setup(
    name = 'pyMilne',
    version = '3.0',
    author = 'J. de la Cruz Rodriguez (ISP-SU 2018 - 2023)',
    ext_modules=[extension],
    cmdclass = {'build_ext': build_ext}
)

