import importlib

import nengo
import numpy as np
from psyrun import Param

from gosmann_frontiers2016.benchmarks.benchmark import benchmark_memory


pspace = Param(model=['spaun'])
pspace *= Param(backend=['reference', 'ocl_gpu', 'ocl_cpu', 'optimized'])
pspace *= Param(trial=range(5))

min_items = 1
max_jobs = None


def execute(model, backend, trial, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if np.isfinite(v)}
    mod = importlib.import_module('gosmann_frontiers2016.benchmarks.' + model)
    model = getattr(mod, model)(**kwargs)
    return benchmark_memory(model, backend)
