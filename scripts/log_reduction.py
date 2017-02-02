import importlib
import sys

import nengo

from gosmann_frontiers2016.backends import optimized
from gosmann_frontiers2016.utils import activate_direct_mode


if __name__ == '__main__':
    model = sys.argv[1]
    mod = importlib.import_module('gosmann_frontiers2016.benchmarks.' + model)

    neuron_type = 'LIF'
    if len(sys.argv) > 2:
        neuron_type = sys.argv[2]

    with nengo.Config(nengo.Ensemble) as cfg:
        if neuron_type != 'Direct':
            cfg[nengo.Ensemble].neuron_type = getattr(
                nengo.neurons, neuron_type)()
        model = getattr(mod, model)()
    if neuron_type == 'Direct':
        activate_direct_mode(model)

    nengo.log(level='INFO')

    with optimized()(model) as sim:
        pass
