from utils.text.io import log
from utils.time.stopwatch import Stopwatch


class Breadcrumb:

    def __init__(self, debug=True, color=None, time_color='green'):
        self.stopwatches = []
        self.index = 0
        self.status = 'off'

        self.debug = debug
        self.color = color
        self.time_color = time_color

    def _append_stopwatch(self):
        self.stopwatches.append(Stopwatch())

    def _pop_stopwatch(self):
        self.stopwatches.pop()

    def _print_delta(self):
        diff = self.stopwatches[self.index].delta()
        log(f'{diff:.3f}s', debug=self.debug, color=self.time_color, indent=(self.index + 1))

    def _print_msg(self, msg):
        log(msg, debug=self.debug, color=self.color, indent=self.index)

    def log(self, msg):
        if self.status == 'off':
            self._append_stopwatch()
            self.status = 'first'
        elif self.status == 'first':
            self._print_delta()
            self.status = 'on'
        else:
            self._print_delta()

        self._print_msg(msg)

    def report(self):
        while self.index >= 0:
            self.outdent()

    def indent(self):
        self.index += 1
        self.status = 'off'

    def outdent(self):
        self._print_delta()
        self._pop_stopwatch()
        self.status = 'on'
        self.index -= 1
