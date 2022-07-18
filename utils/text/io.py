import json
from pathlib import Path


def log(msg, debug):
    """
    Prints msg only if debug evaluates to True.
    """

    if debug:
        print(msg)


class Colors:
    pref = '\033['
    reset = f'{pref}0m'

    codes = {
        'black': '30m',
        'red': '31m',
        'green': '32m',
        'yellow': '33m',
        'blue': '34m',
        'magenta': '35m',
        'cyan': '36m',
        'white': '37m',
    }


def cprint(text, color='white', is_bold=False):
    """
    Print in a given color.

    Args:
        text (str): Text to print.
        color (str): Color to print in.
        is_bold (bool): Should print in bold case.
    """

    print(f'{Colors.pref}{1 if is_bold else 0};{Colors.codes.get(color, Colors.codes["white"])}' + text + Colors.reset)


def pprint(t, indent=0, inline=False, only_first=False):
    """
    Pretty print dictionary.

    Args:
        t (dict): Dictionary to print.
        indent (int): Number of spaces preceding every line.
        inline (bool): Skip preceding spaces for the first line.
        only_first (bool): If True, prints only an ellipsis to avoid big blocks.
    """

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
    """
    Reads json file.

    Args:
        filename (str): Name of the file to be read.

    Returns:
        dict: Dictionary with the contents of the json file.
    """

    with open(filename) as file:
        return json.load(file)


def save_json(data, filename):
    """
    Saves a dictionary as a json file.

    Args:
        data (dict): Dictionary to save.
        filename (str): Name of the file to be read.
    """

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)


def mkdir(dirname):
    """
    Creates the given directory if it does not exist.
    """

    Path(dirname).mkdir(parents=True, exist_ok=True)
