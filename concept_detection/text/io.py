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

    if inline:
        print(t)
    else:
        print(f'{" " * indent}{t}')

