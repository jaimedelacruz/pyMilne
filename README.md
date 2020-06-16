# pyMilne
A parallel python Milne Eddington synthesis/inversion framework.

This module has a C++ backend that takes care of the number crunching in parallel,
while keeping a convenient python interface. 


## Compilation of the C++ module
These routines require building and solving a sparse linear system of
equations. In order to improve performance we have implemented that
part in a C++ module. This module makes use of the Eigen-3 and FFTW-3 
libraries, which should be in your path. The compilation has been tested
in Linux and Mac systems (with MacPorts).

To compile it simply use:
```
python3 setup.py build_ext --inplace
```

If everything compiles well, you should see a new file called pyMilne.???.so
that should be copied along with MilneEddington.py to your PYTHONPATH folder or
to the folder where you want to execute these routines.

## Usage
We refer to the commented example.py file that is included with the distribution.
We have also prepared an example with a real SST/CRISP dataset that can be found [here](https://dubshen.astro.su.se/~jaime/crisp_data/). Simply download all the files included in that folder and run invert_crisp.py. That example is also extensively commented.

## Citing
These routines were developed and used as part of [de la Cruz Rodriguez (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.153D/abstract). If you find these routines useful for your research, I would appreciate it the most if that publication is cited in your paper.
