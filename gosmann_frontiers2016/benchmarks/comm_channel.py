import nengo


def comm_channel(n_neurons=50, dimensions=1, seed=None):
    with nengo.Network(seed=seed, label="Communication channel") as model:
        model.ens_a = nengo.Ensemble(
            n_neurons=n_neurons // 2, dimensions=dimensions)
        model.ens_b = nengo.Ensemble(
            n_neurons=n_neurons // 2, dimensions=dimensions)

        model.stimulus = nengo.Node(0.5)
        nengo.Connection(model.stimulus, model.ens_a)
        nengo.Connection(model.ens_a, model.ens_b)

        model.p_ens_b = nengo.Probe(model.ens_b, synapse=0.01)

    return model
