import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
[line1] = ax.plot([], [], lw=2)
[line2] = ax.plot([], [], lw=2)

n = 60 * 200


def init():
    return [line1, line2]


def animate(i):
    x1 = [0, np.cos(-2 * np.pi * i / 200)]
    y1 = [0, np.sin(-2 * np.pi * i / 200)]
    x2 = [0, 0.5 * np.cos((-2 * np.pi * i / 200) / 60)]
    y2 = [0, 0.5 * np.sin((-2 * np.pi * i / 200) / 60)]
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    return [line1, line2]


anim = FuncAnimation(fig, animate, init_func=init, frames=n, interval=20, blit=True)

plt.show()
