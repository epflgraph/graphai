import math

import matplotlib.pyplot as plt


def bezier(t, points):
    n = len(points)

    coeffs = [(t ** i) * ((1 - t) ** (n - 1 - i)) for i in range(n)]

    x = sum([math.comb(n - 1, i) * coeffs[i] * points[i][0] for i in range(n)])
    y = sum([math.comb(n - 1, i) * coeffs[i] * points[i][1] for i in range(n)])

    return x, y


fig, ax = plt.subplots()

for h in [0, 0.25, 0.5, 0.75, 1]:
    p0 = (0, 0)
    p1 = (1 - h, 0)
    p2 = (0 + h, 1)
    p3 = (1, 1)
    points = [p0, p1, p2, p3]

    path = []
    m = 100
    for t in [i / m for i in range(m + 1)]:
        path.append(bezier(t, points))

    ax.scatter([p[0] for p in path], [p[1] for p in path], s=5)

plt.show()
