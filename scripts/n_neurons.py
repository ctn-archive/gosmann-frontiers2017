import importlib
import sys

import nengo

from gosmann_frontiers2016.backends import optimized
from gosmann_frontiers2016.utils import activate_direct_mode


if __name__ == '__main__':
    model = sys.argv[1]
    mod = importlib.import_module('gosmann_frontiers2016.benchmarks.' + model)

    model = getattr(mod, model)()
    print(sum(e.n_neurons for e in model.all_ensembles))
