import importlib

import nengo
import numpy as np
from psyrun import Param

from gosmann_frontiers2017.benchmarks.benchmark import benchmark_time


pspace = (
    Param(model=['comm_channel']) *
    Param(n_neurons=[100, 200, 500, 1000]) *
    Param(dimensions=[1, 2, 5, 10, 20, 50]))
pspace += (
    Param(model=['lorenz']) *
    Param(n_neurons=[100, 200, 500, 1000, 2000]))
pspace += (
    Param(model=['circ_conv']) *
    Param(n_neurons=[100, 200, 500]) *
    Param(dimensions=[5, 10, 20, 50, 100, 200, 500]))
pspace *= Param(backend=['reference', 'ocl_gpu', 'ocl_cpu', 'optimized'])
pspace *= Param(neuron_type=['LIF', 'LIFRate', 'Direct'])
pspace *= Param(trial=range(5))

min_items = 1
max_jobs = None


def execute(model, backend, neuron_type, trial, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if np.isfinite(v)}
    mod = importlib.import_module('gosmann_frontiers2017.benchmarks.' + model)
    with nengo.Config(nengo.Ensemble) as cfg:
        cfg[nengo.Ensemble].neuron_type = getattr(nengo.neurons, neuron_type)()
        model = getattr(mod, model)(**kwargs)
    return benchmark_time(model, backend)
