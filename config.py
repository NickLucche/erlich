import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from erlich import Erlich, BaseTrainer, AverageEstimator


def create_model_part(architecture_name, cfg, global_cfg):
    # print(architecture_name)
    # print(cfg.pretty())
    return nn.Sequential(
        nn.Conv2d(2, 2, 3),
        nn.ReLU(inplace=True)
    )


class MyTrainer(BaseTrainer):
    def __init__(self, cfg, model_parts, saver, logger, device):
        super().__init__(cfg, model_parts, saver, logger, device)

        self.sino_denoiser = self.model_parts["sino_denoiser"]

    def get_train_metrics(self):
        return AverageEstimator("loss")

    def get_dataloader(self, batch_size):
        img_size = self.cfg.img_size
        input_channels = self.cfg.input_channels
        return DataLoader(TensorDataset(torch.Tensor(64, input_channels, img_size, img_size).normal_()),
                          batch_size=batch_size, pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=4)

    def get_validation_dataloader(self, validation_batch_size):
        img_size = self.cfg.img_size
        input_channels = self.cfg.input_channels
        return DataLoader(TensorDataset(torch.Tensor(64, input_channels, img_size, img_size).normal_()),
                          batch_size=validation_batch_size,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=4)

    def train_step(self, batch, batch_idx, train_metrics):
        x, = batch
        y = self.sino_denoiser(x)
        loss = torch.mean(y)

        train_metrics[0].update(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        return {"validation_loss": 1.0, "ssim": 99.0 + batch_idx}


erlich = Erlich("config", "models", create_model_part)
cfg = erlich.config_from_cli()

trainer = erlich.create_model(MyTrainer, cfg, torch.device("cuda"))
trainer.train(logger_min_wait=0)
