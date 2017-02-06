from __future__ import print_function

import argparse
import importlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Print total neuron number in model.")
    parser.add_argument(
        'model', nargs=1, type=str,
        help="One of the models to be imported "
        "from gosmann_frontiers2016.benchmarks.")
    args = parser.parse_args()

    mod = importlib.import_module(
        'gosmann_frontiers2016.benchmarks.' + args.model[0])

    model = getattr(mod, args.model[0])()
    print(sum(e.n_neurons for e in model.all_ensembles))
