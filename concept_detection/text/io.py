import json

def pprint(t, indent=0, inline=False, only_first=False):
    if type(t) is dict:
        if inline:
            print('{')
        else:
            print(f'{" " * indent}{{')

        for key in t:
            print(f"{' ' * (indent + 2)}'{key}':", end=' ')
            pprint(t[key], indent=indent + 2, inline=True, only_first=only_first)

        print(f'{" " * indent}}}')

        return

    if type(t) is list:
        if inline:
            print('[')
        else:
            print(f'{" " * indent}[')

        for elem in t:
            pprint(elem, indent=indent + 2, only_first=only_first)

            if only_first:
                print(f"{' ' * (indent + 2)}... ({len(t)})")
                break

        print(f'{" " * indent}]')

        return

    # Shorten long strings not to flood the terminal
    if type(t) is str and len(t) > 92:
        t = f'{t[:92]} [...]'

    if inline:
        print(t)
    else:
        print(f'{" " * indent}{t}')


def read_json(filename):
    with open(filename) as file:
        return json.load(file)


class ProgressBar:
    def __init__(self, n_iterations, bar_length=50):
        self.current_iteration = 0
        self.n_iterations = n_iterations
        self.bar_length = bar_length

    def update(self):
        self.current_iteration += 1

        progress = int(self.bar_length * self.current_iteration / self.n_iterations)
        remaining = self.bar_length - progress
        print(f'\r[{"#" * progress}{"." * remaining}]{100 * self.current_iteration / self.n_iterations:.2f}%', end='', flush=True)
