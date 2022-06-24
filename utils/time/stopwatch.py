import time

from utils.text.io import cprint


class Stopwatch:
    def __init__(self):
        self.checkpoints = []
        self.color = 'cyan'
        self.error_color = 'red'

        self.tick()

    def reset(self):
        self.__init__()

    def tick(self):
        self.checkpoints.append(time.time())

    def delta(self):
        self.tick()

        if len(self.checkpoints) < 2:
            return 0

        return self.checkpoints[-1] - self.checkpoints[-2]

    def total(self):
        self.tick()

        if len(self.checkpoints) < 2:
            return 0

        return self.checkpoints[-1] - self.checkpoints[0]

    def report(self, msg='Finished'):
        self.tick()

        n_ticks = len(self.checkpoints)

        if n_ticks < 2:
            cprint(f'ERROR: Stopwatch expected to have at least 2 ticks, only {n_ticks} found.', color=self.error_color)
            return

        cprint(f'{msg}. Total time: {self.checkpoints[-1] - self.checkpoints[0]:.2f}s.', color=self.color)

        if n_ticks == 2:
            return

        for i in range(n_ticks - 1):
            cprint(f'    Lap {i}: {self.checkpoints[i + 1] - self.checkpoints[i]:.2f}s.', color=self.color)
