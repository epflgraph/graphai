import time

from utils.text.io import cprint


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.laps = []
        self.end_time = None
        self.color = 'cyan'

        self.start()

    def start(self):
        self.start_time = time.time()

    def lap(self):
        self.laps.append(time.time())

    def stop(self):
        self.end_time = time.time()

    def delta(self):
        self.stop()
        return self.end_time - self.start_time

    def reset(self):
        self.__init__()

    def report(self, msg=''):
        self.stop()

        if msg:
            cprint(f'{msg}. Took {self.end_time - self.start_time:.2f}s.', color=self.color)
        else:
            cprint(f'Finished! Took {self.end_time - self.start_time:.2f}s.', color=self.color)

        if self.laps:
            self.report_laps()

    def report_laps(self):
        n_laps = len(self.laps)
        for i in range(n_laps + 1):
            if i == 0:
                cprint(f'    Lap 0: {self.laps[0] - self.start_time:.2f}s.', color=self.color)
            elif i == n_laps:
                cprint(f'    Lap {i}: {self.end_time - self.laps[i-1]:.2f}s.', color=self.color)
            else:
                cprint(f'    Lap {i}: {self.laps[i] - self.laps[i-1]:.2f}s.', color=self.color)
