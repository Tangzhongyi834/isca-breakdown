from collections import defaultdict
from contextlib import contextmanager

import torch


class CUDAEventTimer:
    """Deferred event collection; never synchronizes per operator.

    Operator event spans are diagnostic: they may include host submission gaps.
    Actual kernel durations for the paper are obtained from CUPTI correlation.
    """

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.pairs = defaultdict(list)
        self.pending = {}

    def start(self, name):
        if name in self.pending:
            raise RuntimeError(f"Overlapping timer category: {name}")
        with torch.cuda.device(self.device):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
        self.pending[name] = (begin, end)

    def stop(self, name):
        begin, end = self.pending.pop(name)
        with torch.cuda.device(self.device):
            end.record()
        self.pairs[name].append((begin, end))

    @contextmanager
    def measure(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def collect(self, synchronize=True):
        if self.pending:
            raise RuntimeError(f"Unclosed timers: {list(self.pending)}")
        if synchronize:
            torch.cuda.synchronize(self.device)
        return {name: sum(a.elapsed_time(b) for a, b in pairs)
                for name, pairs in self.pairs.items()}
