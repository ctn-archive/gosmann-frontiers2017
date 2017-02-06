import argparse
import importlib

import nengo

from gosmann_frontiers2017.backends import optimized
from gosmann_frontiers2017.utils import activate_direct_mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Log operator reduction in optimizer for given model.")
    parser.add_argument(
        'model', nargs=1, type=str,
        help="One of the models to be imported "
        "from gosmann_frontiers2017.benchmarks.")
    parser.add_argument(
        '--neuron-type', nargs=1, type=str, default=['LIF'],
        help="Neuron type to use.")
    parser.add_argument(
        '--kwargs', nargs=1, type=str, default=['{}'],
        help="Python code returning a dictionary that will be passed as "
        "keyword arguments to the model.")
    args = parser.parse_args()

    mod = importlib.import_module(
        'gosmann_frontiers2017.benchmarks.' + args.model[0])

    with nengo.Config(nengo.Ensemble) as cfg:
        if args.neuron_type[0] != 'Direct':
            cfg[nengo.Ensemble].neuron_type = getattr(
                nengo.neurons, args.neuron_type[0])()
        model = getattr(mod, args.model[0])(**eval(args.kwargs[0], {}))
    if args.neuron_type[0] == 'Direct':
        activate_direct_mode(model)

    nengo.log(level='INFO')

    with optimized()(model) as sim:
        pass
