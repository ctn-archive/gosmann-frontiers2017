import importlib
import time

from nengo.rc import rc


def load_backend(backend):
    mod = importlib.import_module(backend)
    return mod.Simulator


class TimeBlock(object):
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.end = time.time()

    @property
    def duration(self):
        return self.end - self.start


class BenckmarkEnv(object):
    def __init__(self):
        self._decoder_cache = None
        self._progress_bar = None

    def __enter__(self):
        self._decoder_cache = rc.get('decoder_cache', 'enabled')
        self._progress_bar = rc.get('progress', 'progress_bar')
        rc.set('decoder_cache', 'enabled', 'False')
        rc.set('progress', 'progress_bar', 'False')
        return self

    def __exit__(self, exc_type, exc_value, tb):
        rc.set('decoder_cache', 'enabled', self._decoder_cache)
        rc.set('progress', 'progress_bar', self._progress_bar)


def benchmark_time(model, backend='nengo'):
    Simulator = load_backend(backend)

    with BenckmarkEnv():
        with TimeBlock() as t_build:
            sim = Simulator(model)

        try:
            with TimeBlock() as t_prefill:
                sim.run_steps(10)
            with TimeBlock() as t_sim:
                sim.run_steps(1000)
        finally:
            sim.close()

    return {
        't_build': t_build.duration,
        't_prefill': t_prefill.duration,
        't_sim': t_sim.duration,
    }
