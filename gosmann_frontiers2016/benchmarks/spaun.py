try:
    import faulthandler
    faulthandler.enable()
except:
    pass

import nengo
import numpy as np

from gosmann_frontiers2016._spaun.configurator import cfg
from gosmann_frontiers2016._spaun.experimenter import experiment
from gosmann_frontiers2016._spaun.spaun_main import Spaun
from gosmann_frontiers2016._spaun.vocabulator import vocab
from gosmann_frontiers2016._spaun.modules.motor.data import mtr_data
from gosmann_frontiers2016._spaun.modules.vision.data import vis_data


def spaun():
    def_seq = 'A3[123]?XXXX'
    cfg.set_seed(1)
    experiment.initialize(
        def_seq, vis_data.get_image_ind, vis_data.get_image_label,
        cfg.mtr_est_digit_response_time, cfg.rng)
    vocab.initialize(experiment.num_learn_actions, cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)
    return Spaun()


if __name__ == '__main__':
    with nengo.Simulator(spaun()) as sim:
        sim.run(1.)
    print("Success.")
