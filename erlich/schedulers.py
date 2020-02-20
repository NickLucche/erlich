import torch.optim
import numpy as np


class LRScheduler:
    def __init__(self, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.count = 0
        self.lr = 0

    def compute_lr(self, step: int, loss_value: float):
        raise NotImplementedError

    def step(self, loss_value=0.0, size=1):
        self.count += size

        self.lr = self.compute_lr(self.count, loss_value)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        return self.lr


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, lr, warmup_batches=500):
        self.initial_lr = lr
        self.warmup_batches = warmup_batches

        super().__init__(optimizer)

    def compute_lr(self, step: int, loss_value: float):
        # first batches do LR warmup
        if step <= self.warmup_batches:
            return self.initial_lr * (np.exp((step / self.warmup_batches)) - 1) / (np.exp(1) - 1)
        else:
            return self.lr


class WarmupStepScheduler(LRScheduler):
    def __init__(self, optimizer, lr, warmup_batches=500, drop_every=100000, gamma=0.33):
        self.initial_lr = lr
        self.warmup_batches = warmup_batches
        self.drop_every = drop_every
        self.gamma = gamma

        super().__init__(optimizer)

    def compute_lr(self, step: int, loss_value: float):
        # first batches do LR warmup
        if step <= self.warmup_batches:
            return self.initial_lr * (np.exp((step / self.warmup_batches)) - 1) / (np.exp(1) - 1)
        else:
            return self.initial_lr * (self.gamma ** (step // self.drop_every))


class WarmupPlateauScheduler(LRScheduler):
    def __init__(self, optimizer, lr, warmup_batches=500, gamma=0.5, plateau_size=100, plateau_eps=-1e-3, patience=15):
        self.initial_lr = lr
        self.warmup_batches = warmup_batches

        self.gamma = gamma
        self.plateau_size = plateau_size
        self.plateau_eps = plateau_eps
        self.patience = patience
        self.patience_count = 0

        xx = (np.arange(self.plateau_size, dtype=np.float32) - self.plateau_size).reshape(1, -1)
        A = np.vstack((xx, np.ones(self.plateau_size)))
        self.A = np.linalg.pinv(A)
        self.loss_values = []

        super().__init__(optimizer)

    def compute_lr(self, step: int, loss_value: float):
        # accumulate loss values
        self.loss_values.append(loss_value)
        if len(self.loss_values) > self.plateau_size:
            self.loss_values = self.loss_values[-self.plateau_size:]

        # first batches do LR warmup
        if step <= self.warmup_batches:
            return self.initial_lr * (np.exp((step / self.warmup_batches)) - 1) / (np.exp(1) - 1)
        else:
            # if has enough loss values fit a line to see if the loss is decreasing
            if len(self.loss_values) == self.plateau_size:
                x = np.asarray(self.loss_values).reshape(1, -1)
                mb = np.dot(x, self.A)
                m, b = mb[0, 0], mb[0, 1]

                # if loss is not decreasing enough increase patience count
                relative_delta = m / abs(b)
                if relative_delta > self.plateau_eps:
                    self.patience_count += 1
                else:
                    self.patience_count = 0

                # drop the learning rate and reset patience and loss
                if self.patience_count >= self.patience:
                    self.patience_count = 0
                    self.loss_values = []
                    return self.lr * self.gamma

            return self.lr
