import json
import os
import time
from glob import glob
import torch
import torch.nn as nn

from .saver import ModelSaver
from .logging import TrainLogger
from .trainer import BaseTrainer


class ModelManager:
    def __init__(self, folder, model_constructor=None):
        self.folder = folder
        self.model_constructor = model_constructor
        self.models = [self._read(x) for x in glob(os.path.join(folder, "*.json"))]

    @staticmethod
    def _read(path):
        with open(path) as f:
            return json.load(f)

    def create_model(self, arch, params, trainer: BaseTrainer, **kwargs) -> (nn.Module):
        if self.model_constructor is None:
            raise Exception("model_constructor is None\nPlease specify model_constructor function in ModelManager constructor")

        mdl_id = self.get_next_id()
        mdl_path = os.path.join(self.folder, str(mdl_id))
        mdl = self.model_constructor(arch, params)
        logger = TrainLogger(mdl_path + ".log", trainer.epochs)
        saver = ModelSaver(self, mdl_id, arch, params, trainer.batch_size, trainer.optimizers_cfg, trainer.epochs, kwargs)

        trainer.setup(mdl, logger, saver)

        return mdl

    def get_next_id(self):
        self.models.append({})
        return len(self.models) - 1

    def save(self, obj, mid, mdl, current_epoch, validation_metrics, optimizers, amp):
        cfg = {
            "architecture": obj.arch,
            "params": obj.params,
            "batch_size": obj.batch_size,
            "optimizer": obj.optimizer,
            "tot_epochs": obj.tot_epochs,
            "other": obj.other,
            "curr_time": time.time(),
            "current_epoch": int(current_epoch),
            "validation_metrics": validation_metrics
        }

        self.models[mid] = cfg

        print(f"Saving model at {os.path.join(self.folder, f'{mid}')}")
        with open(os.path.join(self.folder, f"{mid}.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        torch.save({
            "model": mdl.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "amp": amp.state_dict() if amp is not None else None
        }, os.path.join(self.folder, f"{mid}.pth"))
