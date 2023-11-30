# pyMilne
A parallel python Milne Eddington synthesis/inversion framework.

This module has a C++ backend that takes care of the number crunching in parallel,
while keeping a convenient python interface. 

The code makes use of analytical derivatives of the emerging intensity (e.g., [Orozco Suarez & del Toro Iniesta 2007](https://ui.adsabs.harvard.edu/abs/2007A%26A...462.1137O)) during the inversion process.

## Compilation of the C++ module
This module makes use of the Eigen-3 (3.3.7) and FFTW-3 
libraries, which should be in your path. The compilation has been tested
in Linux and Mac systems (with MacPorts).

To compile it simply use:
```
python3 setup.py build_ext --inplace
```

If everything compiles well, you should see a new file called pyMilne.???.so
that should be copied along with MilneEddington.py to your PYTHONPATH folder or
to the folder where you want to execute these routines.

NOTE: We need a modern version of Eigen3. If your system does not have it,
simply download the latest stable version and untar it in the pyMilne directory.
Just rename the eigen-3.3.7 folder to eigen3 and the code should compile.

### Using Anaconda python
You can use Anaconda python as a package manager to install all dependencies that are required.
This is particularly convenient in OSX, where the Eigen3, FFTW-3 and python are not installed by default.

To do so, we can create a separate environment to install all packages, in this case the environment is called bla but feel free to replace that with a different name:
```
conda create --name bla
conda activate bla
conda install fftw clangxx_osx-64 eigen ipython matplotlib numpy cython scipy astropy llvm-openmp

```
After that, you will be able to compile the binary as explained above. Just remember to load this environment every time you want to use this module.


If you want to use anaconda python in a similar way in Linux, you can follow a very similar approach,
but replacing the compiler packages for gcc:
```
conda create --name bla
conda activate bla
conda install fftw gxx_linux-64 eigen ipython matplotlib numpy cython scipy astropy
```


## Usage
We refer to the commented example.py file that is included with the distribution.
We have also prepared an example with a real SST/CRISP dataset that can be found in the example_CRISP/ folder. Simply run invert_crisp.py. That example is also extensively commented.
We also have included an example that makes use of the spatially-regularized Levenberg-Marquardt (invert_crisp_spatially_regularized.py).

## Citing
These routines were developed and used as part of the study by [de la Cruz Rodriguez (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.153D/abstract). If you find these routines useful for your research, I would appreciate it the most if that publication is cited in your paper.

## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (SUNMAG, grant agreement 759548) and the Horizon Europe programme (MAGHEAT, grant agreement 101088184). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
