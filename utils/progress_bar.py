class ProgressBar:
    def __init__(self, n_iterations, bar_length=50):
        self.current_iteration = 0
        self.n_iterations = n_iterations
        self.bar_length = bar_length

    def update(self):
        self.current_iteration += 1

        progress = int(self.bar_length * self.current_iteration / self.n_iterations)
        remaining = self.bar_length - progress
        print(f'\r[{"#" * progress}{"." * remaining}] {100 * self.current_iteration / self.n_iterations:.2f}%', end='', flush=True)

        if self.current_iteration == self.n_iterations:
            print()
