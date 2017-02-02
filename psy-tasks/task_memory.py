import importlib

import numpy as np
from psyrun import Param

from gosmann_frontiers2017.benchmarks.benchmark import benchmark_memory


pspace = (
    Param(model=['circ_conv']) *
    Param(n_neurons=[500]) *
    Param(dimensions=[500]))
pspace += Param(model=['nback'])
pspace *= Param(backend=['reference', 'ocl_gpu', 'ocl_cpu', 'optimized'])
pspace *= Param(trial=range(5))

min_items = 1
max_jobs = None


def execute(model, backend, trial, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if np.isfinite(v)}
    mod = importlib.import_module('gosmann_frontiers2017.benchmarks.' + model)
    model = getattr(mod, model)(**kwargs)
    return benchmark_memory(model, backend)
