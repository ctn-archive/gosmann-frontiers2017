try:
    import faulthandler
    faulthandler.enable()
except:
    pass

import nengo
import numpy as np

import gosmann_frontiers2017.spaopt as spa
from gosmann_frontiers2017.spaopt import CircularConvolution


_trial_duration = 2.5
_nback_seq = 'SRVSLVVRHPWWMPWMBGVBLFPLFSLRKMMTNMTKMMKMBBSZBPKB'
SIM_DURATION = _trial_duration * len(_nback_seq) + 0.01


def nback(sd=16):
    symbols = ['BCDFGHJKLMNPQRSTVWXZ']
    n = 3
    d = 64
    neurons_per_dimension = 50
    seed = 1

    vocab = spa.Vocabulary(d)
    vocab.add('Ctx', vocab.create_pointer(unitary=True))
    for s in symbols:
        vocab.parse(s)
    conf = {
        'dimensions': d,
        'subdimensions': sd,
        'neurons_per_dimension': neurons_per_dimension,
        'vocab': vocab
    }

    state_vocab = spa.Vocabulary(d)
    state_vocab.parse('Encode + Wait + Transfer')
    state_conf = {
        'dimensions': d,
        'subdimensions': sd,
        'neurons_per_dimension': neurons_per_dimension,
        'vocab': state_vocab
    }


    def inhibit(pre, post):
        for e in post.ensembles:
            nengo.Connection(
                pre, e.neurons, transform=[[-5]] * e.n_neurons, synapse=tau_gaba)


    class Control(object):
        encode_duration = 0.5

        def trial_t(self, t):
            return t % _trial_duration

        def stim_in(self, t):
            i = int(t // _trial_duration)
            if i >= len(_nback_seq) or self.trial_t(t) > self.encode_duration:
                return '0'
            return _nback_seq[i]

        def cue(self, t):
            return '*'.join(n * ['Ctx'])


    class NBack(spa.SPA):
        def __init__(self, seed):
            super(NBack, self).__init__(seed=seed)
            x = 1.0 / np.sqrt(n)

            with self:
                self.ctrl = Control()

                self.state = spa.Buffer(**conf)

                self.stim_in = spa.Buffer(**conf)
                self.stim_gate = spa.Buffer(**conf)
                self.stim = spa.Memory(synapse=0.1, **conf)

                self.wm_in = spa.Buffer(**conf)
                self.gate1 = spa.Buffer(**conf)
                self.wm1 = spa.Memory(synapse=0.1, **conf)

                self.gate2 = spa.Buffer(**conf)
                self.wm2 = spa.Memory(synapse=0.1, **conf)

                self.gate3 = spa.Buffer(**conf)
                self.wm3 = spa.Memory(synapse=0.1, **conf)

                self.cue = spa.Buffer(**conf)
                self.cc = CircularConvolution(
                    n_neurons=200, dimensions=d, invert_b=True)
                self.comp = spa.Compare(
                    dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                    vocab=vocab)

                self.response = nengo.Ensemble(neurons_per_dimension, 1)
                nengo.Connection(self.response, self.response, synapse=0.1)
                self.rectify = nengo.Ensemble(
                    neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(-0., 1),
                    encoders=nengo.dists.Choice([[1]]))
                nengo.Connection(
                    self.comp.output, self.rectify,
                    transform=[2 * vocab.parse('YES').v])
                nengo.Connection(self.rectify, self.response,
                    function=lambda y: np.maximum(0, y))
                nengo.Connection(nengo.Node(output=-np.exp(-n / 0.62) - 0.2), self.response)

                self.stim_in_dot = spa.Compare(
                    dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                    vocab=vocab)
                nengo.Connection(
                    self.stim_in.state.output, self.stim_in_dot.inputA)
                nengo.Connection(
                    self.stim_in.state.output, self.stim_in_dot.inputB)
                self.wm1_dot = spa.Compare(
                    dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                    vocab=vocab)
                nengo.Connection(self.wm1.state.output, self.wm1_dot.inputA)
                nengo.Connection(self.gate1.state.output, self.wm1_dot.inputB)
                self.wm2_dot = spa.Compare(
                    dimensions=d, neurons_per_multiply=2 * neurons_per_dimension,
                    vocab=vocab)
                nengo.Connection(self.wm2.state.output, self.wm2_dot.inputA)
                nengo.Connection(self.gate1.state.output, self.wm2_dot.inputB)

                self.bg = spa.BasalGanglia(spa.Actions(
                    '0.2 --> state = Encode',
                    'dot(state, Encode) + dot(state, Wait) --> state = Wait',
                    'dot(state, Transfer) --> state = Transfer',
                ))
                self.thalamus = spa.Thalamus(self.bg)

                nengo.Connection(
                    self.stim_in_dot.output, self.bg.input[0],
                    transform=[vocab.parse('YES').v])
                inhibit(self.thalamus.output[0], self.gate2.state)
                nengo.Connection(
                    self.thalamus.output[0], self.response.neurons,
                    transform=[[-5]] * neurons_per_dimension, synapse=tau_gaba)
                inhibit(self.thalamus.output[1], self.stim_gate.state)
                inhibit(self.thalamus.output[1], self.gate1.state)
                inhibit(self.thalamus.output[1], self.gate3.state)
                self.e1 = nengo.Ensemble(neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(0.4, 1), encoders=nengo.dists.Choice([[1]]))
                self.e2 = nengo.Ensemble(neurons_per_dimension, 1, intercepts=nengo.dists.Uniform(0.8, 1), encoders=nengo.dists.Choice([[-1]]))
                nengo.Connection(self.response, self.e1)
                nengo.Connection(self.response, self.e2)
                nengo.Connection(self.e1, self.bg.input[2], function=lambda x: 1 if x > 0.5 else 0)
                nengo.Connection(self.e2, self.bg.input[2], function=lambda x: 1 if x < -0.9 else 0)
                inhibit(self.thalamus.output[2], self.gate1.state)
                inhibit(self.thalamus.output[2], self.stim_gate.state)
                nengo.Connection(
                    self.thalamus.output[2], self.response.neurons,
                    transform=[[-5]] * neurons_per_dimension, synapse=tau_gaba)

                y = np.sqrt((n - 1) * x * x)
                self.cortical1 = spa.Cortical(spa.Actions(
                    'wm_in = {x} * stim_in + {y} * wm2'.format(
                        x=x, y=y),
                    'gate1 = wm_in - wm1',
                    'wm1 = 3 * gate1',
                    'gate2 = wm1 * Ctx - wm2',
                    'wm2 = gate2',
                    'gate3 = wm2 - wm3',
                    'wm3 = 3 * gate3',
                    'stim_gate = stim_in - stim',
                    'stim = stim_gate',), synapse=0.005)

                nengo.Connection(self.wm3.state.output, self.cc.A)
                nengo.Connection(self.cue.state.output, self.cc.B)
                nengo.Connection(self.stim.state.output, self.comp.inputA)
                nengo.Connection(self.cc.output, self.comp.inputB)

                self.input = spa.Input(
                    stim_in=self.ctrl.stim_in, cue=self.ctrl.cue)


    tau_gaba = 0.00848
    model = NBack(seed=seed)
    with model:
        comp_out = nengo.Node(size_in=1)
        nengo.Connection(
            model.comp.output, comp_out, synapse=0.005,
            transform=[vocab.parse('YES').v])

    return model


if __name__ == '__main__':
    with nengo.Simulator(nback()) as sim:
        sim.run(1.)
    print("Success.")
