import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


losses = []
with open("log") as f:
    for line in f:
        loss = float(line.split("    ")[2].strip().split(" ")[0])
        losses.append(loss)

losses = np.array(losses)
plt.plot(losses)
plt.plot(moving_average(losses, 100))
plt.show()
