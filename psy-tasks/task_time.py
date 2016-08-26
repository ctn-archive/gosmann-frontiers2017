import importlib

from psyrun import Param

from gosmann_frontiers2016.benchmarks.benchmark import benchmark_time


pspace = Param(model=['comm_channel', 'lorenz', 'circ_conv'])
pspace *= Param(backend=['reference', 'ocl_gpu', 'ocl_cpu'])
pspace *= Param(trial=range(5))


def execute(model, backend, trial):
    mod = importlib.import_module('gosmann_frontiers2016.benchmarks.' + model)
    model = getattr(mod, model)()
    return benchmark_time(model, backend)
