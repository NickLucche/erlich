import torch
from tqdm import tqdm
import abc
from torch.nn.parallel import DistributedDataParallel
from .schedulers import WarmupScheduler, WarmupPlateauScheduler, WarmupStepScheduler

try:
    from torch.cuda import amp
    print("[Erlich INFO] Imported mixed precision from torch (set 'mixed_precision: true' in config to use it)")
    HAS_APEX = True
except ImportError:
    try:
        from apex import amp
        print("[Erlich INFO] Imported mixed precision from apex (set 'mixed_precision: true' in config to use it)")
        HAS_APEX = True
    except ImportError:
        print("[Erlich INFO] AMP not found, mixed precision training not available")
        amp = None
        HAS_APEX = False


def move_to_device(batch, device):
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [x.to(device) for x in batch]
    else:
        return batch.to(device)


def get_batch_size(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        return batch[0].size(0)
    else:
        return batch.size(0)


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
    def __init__(self, cfg, model_parts, saver, logger, device, rank=0, world_size=1):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.validation_batch_size = cfg.validation_batch_size
        self.epochs = cfg.epochs
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.optimizers = dict()
        self.schedulers = dict()
        self.model_parts = model_parts
        self.saver = saver
        self.logger = logger

        self.dataloader = None
        self.validation_dataloader = None
        self.model = None

        self.train_metrics = self.get_train_metrics()

        # Add train metrics to logger
        if self.logger is not None:
            for metric in self.train_metrics.values():
                self.logger.add_meter(metric)

    def create_dataloaders(self):
        self.dataloader = self.get_dataloader(self.batch_size)
        self.validation_dataloader = self.get_validation_dataloader(self.validation_batch_size)

    @staticmethod
    def standardize_kwargs(cfg, **kwargs):
        return {k: cfg[k] if k in cfg else kwargs[k] for k in kwargs}

    def default_get_scheduler(self, name, optimizer, sched_cfg, _):
        if name == "warmup_plateau":
            cfg = self.standardize_kwargs(sched_cfg, lr=0.1, warmup_batches=500, gamma=0.5, plateau_size=100,
                                          plateau_eps=-1e-3,
                                          patience=15)
            print("Scheduler cfg", cfg)
            return WarmupPlateauScheduler(optimizer, **cfg)
        elif name == "warmup":
            cfg = self.standardize_kwargs(sched_cfg, lr=0.1, warmup_batches=500)
            print("Scheduler cfg", cfg)
            return WarmupScheduler(optimizer, **cfg)
        elif name == "warmup_step":
            cfg = self.standardize_kwargs(sched_cfg, lr=0.1, warmup_batches=500, drop_every=100000, gamma=0.33)
            print("Scheduler cfg", cfg)
            return WarmupStepScheduler(optimizer, **cfg)
        else:
            raise Exception(f"Unknown scheduler '{name}', please override 'get_scheduler' method to add this scheduler")

    def get_scheduler(self, name, optimizer, sched_cfg, cfg):
        return self.default_get_scheduler(name, optimizer, sched_cfg, cfg)

    def default_get_optimizer(self, name, optim_cfg, parameters, _):
        if name == "adam":
            cfg = self.standardize_kwargs(optim_cfg, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0, amsgrad=False)
            return torch.optim.Adam(parameters, **cfg)
        if name == "sgd":
            cfg = self.standardize_kwargs(optim_cfg, lr=1e-3, momentum=0, dampening=0,
                                          weight_decay=0, nesterov=False)
            return torch.optim.SGD(parameters, **cfg)
        if name == "adamW":
            cfg = self.standardize_kwargs(optim_cfg, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0.01, amsgrad=False)
            return torch.optim.AdamW(parameters, **cfg)
        else:
            raise Exception(f"Unknown optimizer '{name}', please override 'get_optimizer' method to add this optimizer")

    def get_optimizer(self, name, optim_cfg, parameters, cfg):
        return self.default_get_optimizer(name, optim_cfg, parameters, cfg)

    def instantiate_optimizers(self, cfg):
        require_global_optimizer = []
        for part_name in cfg.parts:
            part = cfg.parts[part_name]
            # if part requires an optimizer
            if "frozen" not in part or part["frozen"] is False:
                # if part has specific optimizer
                if "optimizer" in part:
                    self.optimizers[part_name] = self.get_optimizer(part.optimizer.name, part.optimizer,
                                                                    self.model_parts[part_name].parameters(), cfg)
                    if "scheduler" in part.optimizer:
                        sched = part.optimizer.scheduler
                        self.schedulers[part_name] = self.get_scheduler(sched.name, self.optimizers[part_name], sched,
                                                                        cfg)
                else:
                    require_global_optimizer.append(part_name)

        # if some parts require the global optimizer instantiate it
        if require_global_optimizer:
            global_opt_parameters = []
            for x in require_global_optimizer:
                global_opt_parameters += list(self.model_parts[x].parameters())

            self.optimizers["__global"] = self.get_optimizer(cfg.optimizer.name, cfg.optimizer,
                                                             global_opt_parameters, cfg)

            if "scheduler" in cfg.optimizer:
                sched = cfg.optimizer.scheduler
                self.schedulers["__global"] = self.get_scheduler(sched.name, self.optimizers["__global"], sched, cfg)

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

    def get_train_metrics(self):
        return dict()

    def pack_model(self):
        return None

    def validate(self, epoch, train_batch, use_apex):
        if self.validation_dataloader is not None:
            print("Validating model")
            self.before_validation(epoch, train_batch)
            estimators = dict()
            for batch_idx, batch in enumerate(tqdm(self.validation_dataloader)):
                with torch.no_grad():
                    batch = move_to_device(batch, self.device)

                    metrics = self.validation_step(batch, batch_idx)
                    # by default weight by batch size
                    w = float(metrics.get("weight", get_batch_size(batch)))
                    for k in metrics:
                        if k not in estimators:
                            estimators[k] = AvgEstimator()
                        estimators[k].update(metrics[k], w)

            estimators = {k: estimators[k].get() for k in estimators}
            print(estimators)
        else:
            estimators = dict()

        if self.saver is not None:
            self.saver.save(self.model_parts, self.optimizers, amp if use_apex else None, epoch, train_batch,
                            estimators)

        # TODO barrier?

    def compute_validate_every(self, validate_every=-1):
        # Define the set of batches IDs after which model is validated
        if validate_every == -1:
            return set()
        else:
            # exclude last batch because it is validated on epoch
            return {i for i in range(validate_every, len(self.dataloader), validate_every)}.difference(
                {len(self.dataloader) - 1})

    def init_apex(self, num_losses=1):
        optimization_level = self.cfg.apex
        if self.rank == 0:
            print("")
            print("=" * 20, "APEX", "=" * 20)

        # Convert dicts to lists
        part_keys = sorted(list(self.model_parts.keys()))
        opt_keys = sorted(list(self.optimizers.keys()))
        parts = [self.model_parts[k] for k in part_keys]
        optimizers = [self.optimizers[k] for k in opt_keys]

        parts, optimizers = amp.initialize(parts, optimizers, opt_level=optimization_level,
                                           verbosity=1 if self.rank == 0 else 0, num_losses=num_losses)

        # convert back to dicts
        self.model_parts = {k: x for k, x in zip(part_keys, parts)}
        self.optimizers = {k: x for k, x in zip(opt_keys, optimizers)}

    def init_training(self, validate_every=-1, logger_min_wait=5, distributed_data_parallel=False, num_losses=1):
        # Define the set of batches IDs after which model is validated
        validate_every = self.compute_validate_every(validate_every)

        if self.logger is not None:
            self.logger.min_wait = logger_min_wait

        using_apex = self.cfg.get("mixed_precision", False) and HAS_APEX
        if using_apex:
            self.init_apex(num_losses)

        if distributed_data_parallel:
            self.model = self.pack_model()
            if self.model is None:
                raise Exception("When using distributed training you need to implement `pack_model`")
            print("Initializing DistributedDataParallel")
            self.model = DistributedDataParallel(self.model, device_ids=[self.device], output_device=self.device)

        if self.rank == 0:
            print("\n")
            print("=" * 20, "TRAINING", "=" * 20)

        if self.logger is not None:
            self.logger.start(self.dataloader)

        return validate_every, using_apex

    def train(self, validate_every=-1, logger_min_wait=5, distributed_data_parallel=False):
        self.before_training()
        validate_every, using_apex = self.init_training(validate_every, logger_min_wait, distributed_data_parallel, 1)

        for epoch in range(self.epochs):
            self.before_train_epoch(epoch)
            for batch_idx, batch in enumerate(self.dataloader):
                # zero grad
                for optim in self.optimizers.values():
                    optim.zero_grad()

                # move data to device
                batch = move_to_device(batch, self.device)

                # do forward step
                loss = self.train_step(batch, batch_idx, self.train_metrics)

                if not isinstance(loss, float):
                    if using_apex:
                        with amp.scale_loss(loss, list(self.optimizers.values())) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    for optim in self.optimizers.values():
                        optim.step()

                    for name in self.schedulers:
                        self.schedulers[name].step(loss.item())

                if self.logger is not None:
                    self.logger.batch()

                # TODO split validation across nodes
                if batch_idx in validate_every and self.rank == 0:
                    self.validate(epoch, batch_idx, using_apex)

            if self.logger is not None:
                self.logger.epoch()

            # TODO split validation across nodes
            if self.rank == 0:
                self.validate(epoch, len(self.dataloader), using_apex)

    def before_training(self):
        pass

    def before_train_epoch(self, epoch: int):
        pass

    def before_validation(self, epoch: int, train_batch: int):
        pass
