# Automatic Optimization of the Computation Graph in the Nengo Neural Network Simulator

One critical factor limiting the size of neural cognitive models is the time
required to simulate such models. To reduce simulation time, specialized
hardware is often used. However, such hardware can be costly, not readily
available, or require specialized software implementations that are difficult to
maintain. Here, we present an algorithm that optimizes the computational graph
of the Nengo neural network simulator, allowing simulations to run more quickly
on commodity hardware.  This is achieved by merging identical operations into
single operations and restructuring the accessed data in larger blocks of
sequential memory. In this way, a time speed-up of up to 6.8 is obtained. While
this does not beat the specialized OpenCL implementation of Nengo, this
optimization is available on any platform that can run Python.  In contrast, the
OpenCL implementation supports fewer platforms and can be difficult to install.

## Requirements

To run the source code [Python](https://www.python.org) is required. Benchmarks
were run with Python 3.4.2, while data analysis and plotting code was run with
Python 3.6.1. The code might run as well with Python 2.7, but this was not
tested.

The complete list of dependencies to run all parts is given in
`requirements.txt` with the exact versions used.

## Installation

It is best to use a newly created [virtualenv](https://virtualenv.pypa.io/en/stable/) for the installation.

1. `git clone --recursive https://github.com/ctn-archive/gosmann-frontiers2017.git`
2. `cd gosmann-frontiers2017`
3. `pip install -r requirements.txt`
4. `pip install .`
5. `cd spaun2.0/_spaun/arms/three_link`
6. `pip install .`

## Reproducing results

### Benchmarks

To run the benchmarks run `psy-doit` from the root folder of the project. This
will take quite some time (up to a few days). The data will be stored in the
files

* `psy-work/memory/result.npz`,
* `psy-work/memory_spaun/result.npz`,
* `psy-work/time_cconv/result.npz`,
* `psy-work/time_nback/result.npz`,
* `psy-work/time_spaun/result.npz`.

Precomputed `time_*/result.npz` data files can be found in the `data` folder.
The memory data files are not included due to their larger size. To just
generate the memory data files you can run `psy-doit memory memory_spaun`.

### Operator reduction

The reduction in operators for a model can be printed with `python
scripts/log_reduction.py <model>` where model can be one of `circ_conv`,
`lorenz`, `nback`, or `spaun`. This script supports additional arguments like
`--neuron-type`. A list of all arguments can be printed with `python
scripts/log_reduction.py -h`.

The `data` folder contains text files with the output for different models with
different neurons types.

### Total number of neurons

The total number of neurons in a model can be printed with `python
scripts/n_neurons <model>`.

### Figures and analysis

The analysis and plotting code is contained in [Jupyter
notebooks](http://www.jupyter.org) and can be opened with:

1. `cd notebooks`
2. `jupyter notebook`

The notebooks require the corresponding data files in the `data` folder. The
time data is contained in this repository, but the memory data needs to be
generated first and copied to that directory.
