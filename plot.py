import matplotlib.pyplot as plt
import json
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

losses = []
xs = []
with open("12.log") as f:
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
plt.plot(x, losses)
plt.plot(x[:-24], moving_average(losses, 25))
plt.show()
"""
a = []
b = []
with open("0.log") as f:
    for line in f:
        obj = json.loads(line)
        loss = obj["Loss"]
        a.append(loss)
with open("1.log") as f:
    for line in f:
        obj = json.loads(line)
        loss = obj["Loss"]
        b.append(loss)
plt.plot(np.log(a))
plt.plot(np.log(b))
plt.show()
"""
