import importlib

from psyrun import Param

from gosmann_frontiers2016.benchmarks.benchmark import benchmark_memory


pspace = Param(model=['comm_channel', 'lorenz', 'circ_conv'])
pspace *= Param(backend=['nengo'])
pspace *= Param(trial=range(5))


def execute(model, backend, trial):
    mod = importlib.import_module('gosmann_frontiers2016.benchmarks.' + model)
    model = getattr(mod, model)()
    return benchmark_memory(model, backend)
