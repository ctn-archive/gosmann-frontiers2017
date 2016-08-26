import importlib
import threading
import time

from nengo.rc import rc
import psutil

from gosmann_frontiers2016 import backends


def load_backend(backend):
    return getattr(backends, backend)()

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


class RecordMemUsage(object):
    def __init__(self, interval=0.01):
        self.interval = interval
        self._start = None
        self.time = []
        self.memory = []
        self.event_times = []
        self._exit = False
        self._process = None

    def _time(self):
        return time.time() - self._start

    def _sample(self):
        self.time.append(self._time())
        self.memory.append(self._process.memory_info().vms)

    def event(self):
        self.event_times.append(self._time())

    def __enter__(self):
        self._process = psutil.Process()

        self._start = time.time()
        self._exit = False

        def target():
            while not self._exit:
                self._sample()
                time.sleep(self.interval)

        threading.Thread(target=target).start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._exit = True
        self._sample()


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


def benchmark_memory(model, backend='nengo'):
    Simulator = load_backend(backend)

    with BenckmarkEnv():
        with RecordMemUsage() as mem_record:
            sim = Simulator(model)
            with sim:
                mem_record.event()
                sim.run_steps(1000)
                mem_record.event()

    return {
        'time': mem_record.time,
        'memory': mem_record.memory,
        'event_times': mem_record.event_times,
    }
