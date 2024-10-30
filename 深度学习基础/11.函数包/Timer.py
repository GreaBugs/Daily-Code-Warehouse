import time


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
        
        
if __name__ == "__main__":
    # Create a Stopwatch instance and start it
    stopwatch = Stopwatch(start=True)

    # Simulate a task by sleeping for a few seconds
    print("Task 1: Starting...")
    time.sleep(2)  # Simulates a task that takes 2 seconds
    elapsed_time = stopwatch.toc()
    print(f"Task 1 completed in {elapsed_time:.2f} seconds.")

    # Accumulate time for a second task
    print("Task 2: Starting...")
    time.sleep(3)  # Simulates a task that takes 3 seconds
    accumulated_time = stopwatch.acc()
    print(f"Task 2 completed in {accumulated_time:.2f} seconds. Total accumulated time: {stopwatch.duration:.2f} seconds.")

    # Restart the stopwatch for a new measurement
    print("Restarting Stopwatch...")
    stopwatch.reset()

    # Simulate another task
    print("Task 3: Starting...")
    time.sleep(1.5)  # Simulates a task that takes 1.5 seconds
    elapsed_time = stopwatch.toc()
    print(f"Task 3 completed in {elapsed_time:.2f} seconds.")

    # Final accumulated time
    print(f"Total accumulated time after reset: {stopwatch.duration:.2f} seconds.")
