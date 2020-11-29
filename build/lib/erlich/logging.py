import time
import json

import torch


class Counter:
    def __init__(self, name, last_value):
        self.name = name

        self.c = 0
        self.last_value = last_value

        self.len = max(len(f"{last_value}/{last_value}"), len(name))

    def increment(self):
        self.c += 1

    def header(self):
        pad = self.len - len(self.name)
        return self.name + " " * pad

    def reset(self):
        self.c = 0

    def to_str(self, curr_epoch, curr_batch):
        res = f"{self.c}/{self.last_value}"
        return res + " " * (self.len - len(res))


class AverageEstimator:
    def __init__(self, name, fmt='{:.2e}'):
        self.name = name
        self.fmt = fmt

        self.len = max(len(self.fmt.format(1.23456e-10)) * 2 + len(" ()"), len(name))

        self.val = 0.0
        self.count = 0.0

        self.epoch_val = 0.0
        self.epoch_count = 0.0

    def header(self):
        pad = self.len - len(self.name)
        return self.name + " " * pad

    def reset(self):
        self.val = 0.0
        self.count = 0.0

    def reset_epoch(self):
        self.epoch_val = 0.0
        self.epoch_count = 0.0

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()

        self.val += x
        self.count += 1

        self.epoch_val += x
        self.epoch_count += 1

    def get_current_value(self):
        return self.val / max(self.count, 1)

    def to_str(self, curr_epoch, curr_batch):
        if self.count > 0:
            mean = self.fmt.format(self.get_current_value())
            epoch_mean = self.fmt.format(self.epoch_val / self.epoch_count)
        else:
            mean = "ND"
            epoch_mean = "ND"

        res = f"{mean} ({epoch_mean})"
        return res + " " * (self.len - len(res))


def fmt_time(s):
    hours = s // 3600
    s = s - (hours * 3600)
    minutes = s // 60
    seconds = s - (minutes * 60)

    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


class TimeEstimator:
    def __init__(self, batches, epochs):
        self.batches = batches
        self.epochs = epochs

        self.start_time = 0

    def start(self):
        self.start_time = time.time()

    def to_str(self, curr_epoch, curr_batch):
        elapsed_time = time.time() - self.start_time
        total_batches = self.epochs * self.batches

        processed_batches = (curr_epoch - 1) * self.batches + curr_batch

        mean_time_per_batch = elapsed_time / processed_batches
        remaining_time = mean_time_per_batch * (total_batches - processed_batches)
        remaining_time_epoch = mean_time_per_batch * (self.batches - curr_batch)

        speed = f"{mean_time_per_batch:.2f} s/b" if mean_time_per_batch > 1.0 else f"{(1.0 / mean_time_per_batch):.2f} b/s"

        return f"{fmt_time(elapsed_time)} < {fmt_time(remaining_time_epoch)} << {fmt_time(remaining_time)}, {speed}"


class TrainLogger:
    def __init__(self, log_path, n_epochs):
        self.log_path = log_path
        self.n_epochs = n_epochs
        self.n_batches = -1
        self.meters = []
        self.min_wait = 5

        self.epoch_counter = Counter("Epoch", n_epochs)
        self.epoch_counter.increment()
        self.batch_counter = None
        self.time = None

        self.last_log_time = 0
        self.train_start_time = -1

        self.log_file = open(self.log_path, "w")

    def add_meter(self, meter):
        self.meters.append(meter)

    def start(self, dataloader):
        self.n_batches = len(dataloader)
        self.batch_counter = Counter("Batch", self.n_batches)

        self.time = TimeEstimator(self.n_batches, self.n_epochs)
        self.time.start()

    def batch(self):
        self.batch_counter.increment()
        t = time.time()

        # don't log last batch because it will be logged by epoch()
        if t - self.last_log_time >= self.min_wait:
            if self.batch_counter.c == 1:
                self.log_headers()
            if self.batch_counter.c < self.n_batches:
                self.log()

                self.last_log_time = t

                for m in self.meters:
                    m.reset()

    def epoch(self):
        self.log()
        print("=" * 80)

        self.batch_counter.reset()
        for m in self.meters:
            m.reset_epoch()

        self.epoch_counter.increment()

    def write_line(self, values):
        data = {
            "epoch": self.epoch_counter.c,
            "batch": self.batch_counter.c,
            "time": time.time()
        }

        for k in values:
            data[k] = values[k]

        self.log_file.write(f"{json.dumps(data)}\n")
        self.log_file.flush()

    def log(self):
        entries = self._base_entries()
        print('    '.join(entries))

        self.write_line({meter.name: meter.get_current_value() for meter in self.meters})

    def _base_entries(self):
        entries = [meter.to_str(self.epoch_counter.c, self.batch_counter.c) for meter in
                   [self.epoch_counter, self.batch_counter] + self.meters + [self.time]]

        return entries

    def log_headers(self):
        print("    ".join([meter.header() for meter in [self.epoch_counter, self.batch_counter] + self.meters]))

    def __del__(self):
        self.log_file.close()
