import matplotlib.pyplot as plt
import json
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


losses = []
xs = []
with open("87.log") as f:
    for line in f:
        obj = json.loads(line)
        loss = obj["Loss"] #float(line.split("    ")[2].strip().split(" ")[0])
        xs.append(obj["time"])
        losses.append(loss)


losses = np.log(np.array(losses))
x = np.array(xs)
x -= np.min(x)
x /= 60
train_time = x[-1] - x[0]
print(f"Train time {train_time//60}h {int(train_time%60)}m")
#plt.plot(x, losses)
plt.plot(x[:-499], moving_average(losses, 500))
plt.show()
"""
a = []
x = []
xx = []
b = []
with open("86.log") as f:
    for line in f:
        obj = json.loads(line)
        loss = obj["Loss"]
        a.append(loss)
        x.append(obj["batch"] + (obj["epoch"]-1)*43496)
with open("84.log") as f:
    for line in f:
        obj = json.loads(line)
        loss = obj["Loss"]
        b.append(loss)
        xx.append(obj["batch"] + (obj["epoch"]-1)*43496)
plt.plot(moving_average(x, 200), moving_average(np.log(a), 200))
plt.plot(moving_average(xx, 200), moving_average(np.log(b), 200))
plt.show()
"""

