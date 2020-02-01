import json
import os
import time
from glob import glob

import torch


class ModelSaver:
    def __init__(self, manager, model_id, arch, params, batch_size, optimizer, tot_epochs, other):
        self.manager = manager
        self.model_id = model_id

        self.arch = arch
        self.params = params
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.tot_epochs = tot_epochs
        self.other = other

    def save(self, mdl, current_epoch, validation_metrics, optimizers, amp):
        self.manager.save(self, self.model_id, mdl, current_epoch, validation_metrics, optimizers, amp)
