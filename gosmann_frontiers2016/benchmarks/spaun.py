try:
    import faulthandler
    faulthandler.enable()
except:
    pass

import nengo
import numpy as np

from gosmann_frontiers2016._spaun.main_main import Spaun


def spaun():
    return Spaun()


if __name__ == '__main__':
    with nengo.Simulator(spaun()) as sim:
        sim.run(1.)
    print("Success.")
