import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from erlich import Erlich, BaseTrainer, AverageEstimator


def create_model_part(name, part_cfg, cfg):
    return nn.Sequential(
        nn.Linear(4, 1),
        nn.ReLU()
    )


class ModelTrainer(BaseTrainer):
    def get_train_metrics(self):
        return AverageEstimator("loss")

    def get_dataloader(self, batch_size):
        return DataLoader(TensorDataset(torch.Tensor(64, 4).normal_()), batch_size=batch_size, pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=4)

    def get_validation_dataloader(self, validation_batch_size):
        return DataLoader(TensorDataset(torch.Tensor(64, 4).normal_()), batch_size=validation_batch_size,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=4)

    def pack_model(self):
        return self.model_parts["sino_denoiser"]

    def train_step(self, batch, batch_idx, train_metrics):
        x, = batch
        y = self.model_parts["sino_denoiser"](x)
        loss = torch.mean(y)

        train_metrics[0].update(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        return {"validation_loss": 1.0, "ssim": 99.0 + batch_idx}


if __name__ == "__main__":
    erlich = Erlich("config", "models", create_model_part)
    cfg = erlich.config_from_cli()
    erlich.train(ModelTrainer, cfg, devices=[0, 0])

# import time
# import pandas as pd
#
# df = pd.read_csv("log.csv")
# print(df)
# print(df[" 0"])
# # with open("log.csv", "a") as f:
# #     for i in range(50):
# #         f.write(f"ciao, {i}\n")
# #         #f.flush()
# #         #time.sleep(1)
