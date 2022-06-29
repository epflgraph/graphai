class ProgressBar:
    """
    Class representing a progress bar to keep track of execution progress. Usage is as follows:

    >>> letters = ['a', 'b', 'c', 'd']
    >>> pb = ProgressBar(len(letters))
    >>> for letter in letters:
    >>>     # Run some time-consuming tasks
    >>>     pb.update()
    [############......................................] 25.00%
    [########################..........................] 50.00%
    [####################################..............] 75.00%
    [##################################################] 100.00%

    This prints at each iteration a progress bar and the percentage of completion, overwriting the previous one.

    The state of progress ban be reset as follows:

    >>> letters = ['a', 'b', 'c', 'd']
    >>> ...
    >>> pb = ProgressBar(len(letters))
    >>> for letter in letters:
    >>>     # Run some time-consuming tasks
    >>>     pb.update()
    >>> ...
    >>> pb.reset()
    >>> for letter in letters:
    >>>     # Run more time-consuming tasks
    >>>     pb.update()
    """

    def __init__(self, n_iterations, bar_length=50):
        self.current_iteration = 0
        self.n_iterations = n_iterations
        self.bar_length = bar_length

    def update(self):
        """
        Increments by one the iterations counter and prints the progress bar.
        """

        self.current_iteration += 1

        progress = int(self.bar_length * self.current_iteration / self.n_iterations)
        remaining = self.bar_length - progress
        print(f'\r[{"#" * progress}{"." * remaining}] {100 * self.current_iteration / self.n_iterations:.2f}%', end='', flush=True)

        if self.current_iteration == self.n_iterations:
            print()

    def reset(self, n_iterations=None):
        """
        Resets the progress bar for reuse.

        Args:
            n_iterations (int): Total number of iterations for completion.
        """

        if n_iterations is None:
            n_iterations = self.n_iterations

        self.__init__(n_iterations)
