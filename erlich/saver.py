import json
import os
import time
from glob import glob

import torch


class ModelSaver:
    def __init__(self, base_path):
        os.mkdir(base_path)
        self.base_path = base_path

    @staticmethod
    def link(link_path, src_path):
        if os.path.exists(link_path):
            os.unlink(link_path)
        os.symlink(src_path, link_path)

    def save(self, parts, optimizers, amp, epoch, batch, metrics):
        path = os.path.join(self.base_path, f"{epoch}.{batch}")
        latest_path = os.path.join(self.base_path, "latest")
        print(f"Saving model checkpoint at '{path}'")

        torch.save({
            "parts": {k: parts[k].state_dict() for k in parts},
            "optimizers": {k: optimizers[k].state_dict() for k in optimizers},
            "amp": amp.state_dict() if amp is not None else None
        }, f"{path}.pth")

        with open(f"{path}.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "batch": batch,
                "time": time.time(),
                "metrics": metrics
            }, f, indent=2)

        # link latest checkpoint for easy reuse
        self.link(f"{latest_path}.json", f"{os.path.basename(path)}.json")
        self.link(f"{latest_path}.pth", f"{os.path.basename(path)}.pth")

# class ModelSaver:
#     def __init__(self, manager, model_id, arch, params, batch_size, optimizer, tot_epochs, other):
#         self.manager = manager
#         self.model_id = model_id
#
#         self.arch = arch
#         self.params = params
#         self.batch_size = batch_size
#         self.optimizer = optimizer
#         self.tot_epochs = tot_epochs
#         self.other = other
#
#     def save(self, mdl, current_epoch, validation_metrics, optimizers, amp):
#         self.manager.save(self, self.model_id, mdl, current_epoch, validation_metrics, optimizers, amp)
