import time

from graphai.core.utils.text import cprint


class Stopwatch:
    """
    Class representing a stopwatch to track execution times. Usage is as follows:

    >>> sw = Stopwatch()
    >>> # Run some time-consuming tasks
    >>> sw.delta()
    4.284306764602661

    Time is returned in seconds and as a float variable. Several consecutive execution blocks can be tracked as well:

    >>> sw = Stopwatch()
    >>> # Run some time-consuming tasks
    >>> t1 = sw.delta()
    >>> # Run more time-consuming tasks
    >>> t2 = sw.delta()
    >>> # Run yet more time-consuming tasks
    >>> t3 = sw.delta()
    >>> print(t1, t2, t3)
    5.609307289123535 2.849977970123291 4.660188913345337

    The total time can also be retrieved:

    >>> sw = Stopwatch()
    >>> # Preprocessing
    >>> pre_time = sw.delta()
    >>> # Run some time-consuming tasks
    >>> task_time = sw.delta()
    >>> # Postprocessing
    >>> post_time = sw.delta()
    >>> total_time = sw.total()
    >>> print(f'Proportion of total time used by task: {task_time/total_time}')
    Proportion of total time used by task: 0.5176954220125477

    The :func:`~utils.time.stopwatch.Stopwatch.tick` function can replace :func:`~utils.time.stopwatch.Stopwatch.delta`
    if the partial time is not needed.

    Furthermore, a report can be printed too:

    >>> sw = Stopwatch()
    >>> # Preprocessing
    >>> sw.tick()
    >>> # Run some time-consuming tasks
    >>> sw.tick()
    >>> # Postprocessing
    >>> sw.report('Completed all tasks')
    Completed all tasks. Total time: 19.86s.
        Lap 0: 6.74s.
        Lap 1: 11.66s.
        Lap 2: 1.45s.

    Finally, a stopwatch can be reset for reuse:

    >>> sw = Stopwatch()
    >>> # Task 1.1
    >>> sw.tick()
    >>> # Task 1.2
    >>> sw.tick()
    >>> # Task 1.3
    >>> sw.report()
    >>> ...
    >>> # Run other tasks without tracking time
    >>> ...
    >>> sw.reset()
    >>> # Task 2.1
    >>> sw.tick()
    >>> # Task 2.2
    >>> sw.report()
    """

    def __init__(self):
        self.checkpoints = []
        self.color = 'cyan'
        self.error_color = 'red'

        self.tick()

    def reset(self):
        """
        Resets all time checkpoints.
        """

        self.__init__()

    def tick(self):
        """
        Creates a new time checkpoint.
        """

        self.checkpoints.append(time.time())

    def delta(self):
        """
        Creates a new time checkpoint and returns the difference with the previous one.
        """

        self.tick()

        if len(self.checkpoints) < 2:
            return 0

        return self.checkpoints[-1] - self.checkpoints[-2]

    def total(self):
        """
        Creates a new time checkpoint and returns the difference with the first one.
        """

        self.tick()

        if len(self.checkpoints) < 2:
            return 0

        return self.checkpoints[-1] - self.checkpoints[0]

    def report(self, tick=False, laps=True, msg='Finished'):
        """
        Prints a summary of all time checkpoints, including the differences between each pair
        of consecutive checkpoints and the difference between the last and the first.

        Args:
            tick (boolean): Whether to create a new time checkpoint before printing.
            msg (str): Message to be printed at the beginning of the summary. Defaults to "Finished".
        """

        if tick:
            self.tick()

        n_ticks = len(self.checkpoints)

        if n_ticks < 2:
            cprint(f'ERROR: Stopwatch expected to have at least 2 ticks, only {n_ticks} found.', color=self.error_color)
            return

        cprint(f'{msg}. Total time: {self.checkpoints[-1] - self.checkpoints[0]:.2f}s.', color=self.color)

        if n_ticks == 2 or not laps:
            return

        for i in range(n_ticks - 1):
            cprint(f'    Lap {i}: {self.checkpoints[i + 1] - self.checkpoints[i]:.2f}s.', color=self.color)
