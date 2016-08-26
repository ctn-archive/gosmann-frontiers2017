import nengo


def comm_channel(seed=None):
    with nengo.Network(seed=seed, label="Communication channel") as model:
        model.ens_a = nengo.Ensemble(n_neurons=30, dimensions=1)
        model.ens_b = nengo.Ensemble(n_neurons=30, dimensions=1)

        model.stimulus = nengo.Node(0.5)
        nengo.Connection(model.stimulus, model.ens_a)
        nengo.Connection(model.ens_a, model.ens_b)

        model.p_ens_b = nengo.Probe(model.ens_b, synapse=0.01)

    return model
