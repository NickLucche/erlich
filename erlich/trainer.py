import torch
from tqdm import tqdm
import abc

from .logging import TrainLogger
from .saver import ModelSaver

try:
    from apex import amp

    HAS_APEX = True
except ImportError:
    print("[Erlich INFO] Apex not found")
    amp = None
    HAS_APEX = False


def move_to_device(batch, device):
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [x.to(device) for x in batch]
    else:
        return batch.to(device)


class AvgEstimator:
    def __init__(self):
        self.avg = 0.0
        self.tot_weight = 0.0

    def update(self, x, w=1.0):
        self.tot_weight += w
        self.avg += (x - self.avg) * w / self.tot_weight

    def get(self):
        return self.avg


class BaseTrainer(abc.ABC):
    def __init__(self, batch_size, validation_batch_size, epochs, optimizers_cfg, device):
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.epochs = epochs
        self.optimizers_cfg = optimizers_cfg
        self.device = device

        self.optimizers = None
        self.model = None
        self.logger = None
        self.saver = None
        self.train_metrics = []

        self.dataloader = None
        self.validation_dataloader = None

    @abc.abstractmethod
    def train_step(self, batch, batch_idx, train_metrics):
        pass

    def validation_step(self, batch, batch_idx):
        return dict()

    @abc.abstractmethod
    def get_dataloader(self, batch_size):
        pass

    def get_validation_dataloader(self, validation_batch_size):
        return None

    def get_optimizers(self, model, cfg):
        # TODO
        return [torch.optim.Adam(model.parameters(), weight_decay=0.0, amsgrad=True)]

    def get_train_metrics(self):
        return []

    def setup(self, model, logger: TrainLogger, saver: ModelSaver):
        self.model = model
        self.logger = logger
        self.saver = saver

        self.optimizers = self.get_optimizers(model, self.optimizers_cfg)

        self.dataloader = self.get_dataloader(self.batch_size)
        self.validation_dataloader = self.get_validation_dataloader(self.validation_batch_size)

        self.train_metrics = self.get_train_metrics()
        if not isinstance(self.train_metrics, list):
            self.train_metrics = [self.train_metrics]

        # Add train metrics to logger
        for metric in self.train_metrics:
            self.logger.add_meter(metric)

    def validate(self, epoch, use_apex):
        validation_metrics = dict()
        if self.validation_dataloader is not None:
            print("Validating model")
            estimators = dict()
            for batch_idx, batch in enumerate(tqdm(self.validation_dataloader)):
                with torch.no_grad():
                    batch = move_to_device(batch, self.device)

                    metrics = self.validation_step(batch, batch_idx)
                    w = float(metrics.get("weight", 1.0))
                    for k in metrics:
                        if k not in estimators:
                            estimators[k] = AvgEstimator()
                        estimators[k].update(metrics[k], w)

            estimators = {k: estimators[k].get() for k in estimators}
            print(estimators)

        print("Saving model")
        self.saver.save(self.model, epoch, validation_metrics, self.optimizers, amp if use_apex else None)

    def train(self, validate_every=-1, logger_min_wait=5, use_apex=False, optimization_level="O1"):
        # Define the set of batches IDs after which model is validated
        if validate_every == -1:
            validate_every = set()
        else:
            # exclude last batch because it is validated on epoch
            validate_every = {i for i in range(validate_every, len(self.dataloader), validate_every)}.difference(
                {len(self.dataloader) - 1})

        self.logger.min_wait = logger_min_wait

        print("Moving model to", self.device)
        self.model = self.model.to(self.device)

        use_apex = use_apex and HAS_APEX
        if use_apex:
            self.model, self.optimizers = amp.initialize(self.model, self.optimizers, opt_level=optimization_level)

        self.logger.start(self.dataloader)
        for epoch in range(self.epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                # zero grad
                for optim in self.optimizers:
                    optim.zero_grad()

                # move data to device
                batch = move_to_device(batch, self.device)

                # do forward step
                loss = self.train_step(batch, batch_idx, self.train_metrics)

                if use_apex:
                    with amp.scale_loss(loss, self.optimizers) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                for optim in self.optimizers:
                    optim.step()

                self.logger.batch()

                if batch_idx in validate_every:
                    self.validate(epoch, use_apex)

            self.logger.epoch()
            self.validate(epoch, use_apex)
