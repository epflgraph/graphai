import Levenshtein
import requests
import random


def lev_distance(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    if a[0] == b[0]:
        return lev_distance(a[1:], b[1:])

    l1 = lev_distance(a[1:], b)
    l2 = lev_distance(a, b[1:])
    l3 = lev_distance(a[1:], b[1:])

    return min(1 + l1, 1 + l2, 1 + l3)


def lev_cost(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    if a[0] == b[0]:
        return lev_cost(a[1:], b[1:])

    l1 = lev_cost(a[1:], b)
    l2 = lev_cost(a, b[1:])
    l3 = lev_cost(a[1:], b[1:])

    return min(1 + l1, 1 + l2, 2 + l3)    # Insertion and deletion have cost 1, replacement has cost 2


def lev_ratio(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0

    return 1 - lev_cost(a, b) / (len(a) + len(b))


words = requests.get("https://www.mit.edu/~ecprice/wordlist.10000").content.splitlines()

words_a = random.sample(words, 10)
words_b = random.sample(words, 10)

for a in words_a:
    for b in words_b:
        print('+', end='')
        my_lr = lev_ratio(a, b)
        true_lr = Levenshtein.ratio(a, b)

        if my_lr - true_lr > 1e-4:
            print(a, b)
            print(my_lr)
            print(true_lr)

    print('#')
