import numpy as np
import time

# Code from ray/Rllib
class TimerStat(object):
    """A running stat for conveniently logging the duration of a code block.

    Example:
        wait_timer = TimerStat()
        with wait_timer:
            ray.wait(...)

    Note that this class is *not* thread-safe.
    """

    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    @property
    def mean(self):
        return np.mean(self._samples)

    @property
    def mean_units_processed(self):
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = sum(self._samples)
        if not time_total:
            return 0.0
        return sum(self._units_processed) / time_total




# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)