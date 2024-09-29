# -*- coding: utf-8 -*-
"""The Stopwatch utility."""

import logging
import statistics
import time
from collections import defaultdict
from typing import Dict
from typing import List

import torch
import torch.distributed as dist
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

class Stopwatch:
    """The matlab-style Stopwatch."""

    def __init__(self, start=False):
        """Initialize.

        Args:
          start (bool): whether start the clock immediately.
        """
        self.beg = None
        self.end = None
        self.duration = 0.0

        self.create_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        if start:
            self.tic()

    def tic(self):
        """Start the Stopwatch."""
        self.beg = time.time()
        self.end = None

    def toc(self):
        """Stop the Stopwatch."""
        if self.beg is None:
            raise RuntimeError("Please run tic before toc.")
        self.end = time.time()
        return self.end - self.beg

    def toc2(self):
        """Record duration, and Restart the Stopwatch."""
        delta = self.toc()
        self.tic()
        return delta

    def acc(self):
        """Accumulates the duration."""
        delta = self.toc()
        self.duration += delta
        self.tic()
        return delta

    def reset(self):
        """Reset the whole Stopwatch."""
        self.tic()
        self.duration = 0.0

class ProfileStopwatch(Stopwatch):
    """Stopwatch for profile.

    Examples::

        >>> # Init ProfileStopwatch
        >>> profile_stopwatch = ProfileStopwatch(
                stopwatch_name="ProfileStopwatch",
                barrier_before_stopwatch=True,
            )
        >>> # Code1 to be profile
        >>> time.sleep(1)
        >>> # Record time cost
        >>> profile_stopwatch.record_time_cost("Code1")
        >>> # Code2 to be profile
        >>> time.sleep(2)
        >>> # Record time cost
        >>> profile_stopwatch.record_time_cost("Code2")
        >>> # Log time cost
        >>> profile_stopwatch.log_time_cost()
        >>> # Outputs:
            +------------------+-----------+-----------+
            | ProfileStopwatch | Time Cost |     Ratio |
            +------------------+-----------+-----------+
            |            Total |  3.003961 | 100.0000% |
            |            Code2 |  2.001887 |  66.6416% |
            |            Code1 |  1.002074 |  33.3584% |
            +------------------+-----------+-----------+
        >>> # Reset stopwatch
        >>> profile_stopwatch.reset()
    """

    def __init__(
        self,
        stopwatch_name: str,
        sync_before_stopwatch: bool = True,
        barrier_before_stopwatch: bool = False,
        log_interval: int = 20,
        reset_after_log_time_cost: bool = True,
        **kwargs,
    ):
        """Initialize.

        Args:
            stopwatch_name (str): Stopwatch name.
            sync_before_stopwatch (bool): Whether enable sync before stopwatch.
                Default=True.
            barrier_before_stopwatch (bool): Whether enable barrier before stopwatch.
                Default=False.
            log_interval (int): Log time cost table every log_interval times calling
                self.log_time_cost method. Default=20.
            reset_after_log_time_cost (bool): Whether clear recorded time previously and
                start new record. Default=True.
        """
        super().__init__(**kwargs)
        self._sync_before_stopwatch = sync_before_stopwatch
        self._barrier_before_stopwatch = barrier_before_stopwatch
        self._time_cost: Dict[str, List[float]] = defaultdict(list)
        self._stopwatch_name = stopwatch_name
        self._count = 0
        self._log_interval = log_interval
        self._reset_after_log_time_cost = reset_after_log_time_cost
        self._init_table()

    def _init_table(self):
        self._table = PrettyTable()
        self._table.field_names = [self._stopwatch_name, "Time Cost", "Ratio"]
        self._table.align[self._stopwatch_name] = "r"
        self._table.align["Time Cost"] = "r"
        self._table.align["Ratio"] = "r"

    def tic(self):
        """Start the Stopwatch."""
        if self._sync_before_stopwatch:
            torch.cuda.synchronize()

        if self._barrier_before_stopwatch:
            dist.barrier()

        return super().tic()  # 开始计时

    def toc(self):
        """Stop the Stopwatch."""
        if self._sync_before_stopwatch:
            torch.cuda.synchronize()

        if self._barrier_before_stopwatch:
            dist.barrier()

        return super().toc()

    def record_time_cost(self, time_cost_key: str):
        """Record time cost.

        Args:
            time_cost_key (str): Key of time cost.
        """
        self._time_cost[time_cost_key].append(self.toc2())

    def reset(self):
        """Reset the whole Stopwatch."""
        super().reset()
        self._time_cost: Dict[str, List[float]] = defaultdict(list)
        self._table.clear_rows()

    def log_time_cost(self):
        """Log time cost."""
        self._count += 1
        if self._count % self._log_interval != 0:
            return

        total_time_cost = 0.0
        for time_cost_list in self._time_cost.values():
            total_time_cost += statistics.mean(time_cost_list)

        self._table.add_row(
            [
                "Total",
                f"{total_time_cost:.6f}",
                f"{(total_time_cost/total_time_cost) * 100:.4f}%",
            ]
        )

        for key, time_cost_list in self._time_cost.items():
            time_cost = statistics.mean(time_cost_list)
            self._table.add_row(
                [key, f"{time_cost:.6f}", f"{(time_cost/total_time_cost) * 100:.4f}%"]
            )

        logger.warning(
            "\n%s",
            self._table.get_string(sortby="Time Cost", reversesort=True),
        )

        if self._reset_after_log_time_cost:
            self.reset()

    def __getstate__(self):
        """Pop unpicklable attributes to make itself picklable."""
        self.__dict__.pop("_table", None)
        return self.__dict__

    def __setstate__(self, state):
        """Recover all attributes."""
        self.__dict__.update(state)
        # reinit unpickable
        self._init_table()
