from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy
from distutils import sysconfig
#import numpy.distutils.intelccompiler
import numpy.distutils.ccompiler
import platform as plt
import sys
import pathlib

os.system('rm pyMilne.*.so pyMilne.*.cpp')
p = pathlib.Path(sys.executable)
root_dir = str(pathlib.Path(*p.parts[0:-2]))


if(plt.system() == 'Darwin'):
    #root_dir = '/opt/local/'
    CC = 'clang'
    CXX= 'clang++'
    link_opts = ["-stdlib=libc++","-bundle","-undefined","dynamic_lookup", "-fopenmp"]
else:
    #root_dir = '/usr/'
    CC = 'gcc'
    CXX= 'g++'
    link_opts = ["-shared", "-fopenmp"]

os.environ["CC"] = CC
os.environ["CXX"] = CXX

from distutils import sysconfig
sysconfig.get_config_vars()['CFLAGS'] = ''
sysconfig.get_config_vars()['OPT'] = ''
sysconfig.get_config_vars()['PY_CFLAGS'] = ''
sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
sysconfig.get_config_vars()['CC'] =  CC
sysconfig.get_config_vars()['CXX'] = CXX
sysconfig.get_config_vars()['BASECFLAGS'] = ''
sysconfig.get_config_vars()['CCSHARED'] = ''
sysconfig.get_config_vars()['LDSHARED'] = CC
sysconfig.get_config_vars()['CPP'] = CXX
sysconfig.get_config_vars()['CPPFLAGS'] = ''
sysconfig.get_config_vars()['BLDSHARED'] = ''
sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
sysconfig.get_config_vars()['LDFLAGS'] = ''
sysconfig.get_config_vars()['PY_LDFLAGS'] = ''



comp_flags=['-Ofast','-std=c++14','-march=native','-fPIC','-fopenmp', '-I./src', '-DNDEBUG']

extension = Extension("pyMilne",
                      sources=["pyMilne.pyx"], 
                      include_dirs=["./", root_dir+"/include/",numpy.get_include(),'./eigen3/'],
                      language="c++",
                      extra_compile_args=comp_flags,
                      extra_link_args=link_opts,
                      library_dirs=[root_dir+'/lib/','./'],
                      libraries=['fftw3'])

extension.cython_directives = {'language_level': "3"}

setup(
    name = 'pyMilne',
    version = '1.0',
    author = 'J. de la Cruz Rodriguez (ISP-SU 2020)',
    ext_modules=[extension],
    cmdclass = {'build_ext': build_ext}
)

