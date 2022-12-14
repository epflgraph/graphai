import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(dpi=150)
ax.set_xlim(0, 2)
ax.set_ylim(-2, 2)
[line] = ax.plot([], [], lw=2)

x = np.linspace(0, 2, 1000)
n = 200


def init():
    return [line]


def animate(i):
    y = np.sin(2 * np.pi * (x - 2 * i / n))
    line.set_data(x, y)
    return [line]


anim = FuncAnimation(fig, animate, init_func=init, frames=n, interval=20, blit=True)

plt.show()
