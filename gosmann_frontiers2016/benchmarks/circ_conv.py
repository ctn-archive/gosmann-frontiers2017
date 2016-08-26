import nengo


def circ_conv(seed=None):
    with nengo.Network(seed=seed, label="Circular convolution") as model:
        model.input_a = nengo.Node([-0.21, 0.5, 0.12, 0.06])
        model.ens_a = nengo.Ensemble(n_neurons=512, dimensions=4)
        nengo.Connection(model.input_a, model.ens_a)

        model.input_b = nengo.Node([-0.18, 0.28, 0.18, -0.52])
        model.ens_b = nengo.Ensemble(n_neurons=512, dimensions=4)
        nengo.Connection(model.input_b, model.ens_b)

        model.cconv = nengo.networks.CircularConvolution(
            n_neurons=1032, dimensions=4)
        nengo.Connection(model.ens_a, model.cconv.A)
        nengo.Connection(model.ens_b, model.cconv.B)

        model.result = nengo.Ensemble(n_neurons=512, dimensions=4)
        nengo.Connection(model.cconv.output, model.result)

        model.p_result = nengo.Probe(model.result, synapse=0.02)

    return model
