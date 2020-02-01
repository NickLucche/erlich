import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from erlich import ModelManager, BaseTrainer, AverageEstimator


def create_model(architecture, params):
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

    def train_step(self, batch, batch_idx, train_metrics):
        x, = batch
        y = self.model(x)
        loss = torch.mean(y)

        train_metrics[0].update(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        return {"validation_loss": 1.0, "ssim": 99.0 + batch_idx}


model_manager = ModelManager("models", create_model)

architecture = "sample"
arch_params = dict()

batch_size = 16
validation_batch_size = 32
epochs = 10
trainer = ModelTrainer(batch_size, validation_batch_size, epochs, "adam", torch.device("cuda"))

model = model_manager.create_model(architecture, arch_params, trainer)
trainer.train(validate_every=20000, logger_min_wait=0, use_apex=False)

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
