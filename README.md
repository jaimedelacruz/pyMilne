# pyMilne
A parallel python Milne Eddington synthesis/inversion framework.

This module has a C++ backend that takes care of the number crunching in parallel,
while keeping a convenient python interface. 

The code makes use of analytical derivatives of the emerging intensity (e.g., [Orozco Suarez & del Toro Iniesta 2007](https://ui.adsabs.harvard.edu/abs/2007A%26A...462.1137O)) during the inversion process.

## Compilation of the C++ module
This module makes use of the Eigen-3 and FFTW-3 
libraries, which should be in your path. The compilation has been tested
in Linux and Mac systems (with MacPorts).

To compile it simply use:
```
python3 setup.py build_ext --inplace
```

If everything compiles well, you should see a new file called pyMilne.???.so
that should be copied along with MilneEddington.py to your PYTHONPATH folder or
to the folder where you want to execute these routines.

### Using Anaconda python
You can use Anaconda python as a package manager to install all dependencies that are required.
This is particularly convenient in OSX, where the Eigen3, FFTW-3 and python are not installed by default.

To do so, we can create a separate environment to install all packages, in this case the environment is called bla but feel free to replace that with a different name:
```
conda create --name bla
conda activate bla
conda install fftw clangxx_osx-64 eigen ipython matplotlib numpy cython scipy astropy llvm-openmp

```
You will need to edit one line in setup.py to point to the folder where you have your environment installed. Please make the root_dir variable point to your environment directory. In my case it looks something like this:
```python
if(plt.system() == 'Darwin'):
    root_dir = '/Users/jaime/anaconda3/envs/bla/'
```


## Usage
We refer to the commented example.py file that is included with the distribution.
We have also prepared an example with a real SST/CRISP dataset that can be found [here](https://dubshen.astro.su.se/~jaime/crisp_data/). Simply download all the files included in that folder and run invert_crisp.py. That example is also extensively commented.

## Citing
These routines were developed and used as part of [de la Cruz Rodriguez (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.153D/abstract). If you find these routines useful for your research, I would appreciate it the most if that publication is cited in your paper.
