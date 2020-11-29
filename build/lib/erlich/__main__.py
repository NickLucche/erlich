import os
import glob
import sys
import json
import pandas as pd

from omegaconf import OmegaConf

"""
{
  "epoch": 3,
  "batch": 20000,
  "time": 1581666555.5605233,
  "metrics": {
    "loss": 0.006205861160745297,
    "weight": 47.99483037156704
  }
}
"""

data = []

model_folder = "models"

for path in glob.glob(f"{model_folder}/*.yaml"):
    _, model_id = path[:-5].rsplit("/", 1)

    # TODO handle also best
    latest_path = f"{model_folder}/{model_id}/latest.json"
    if os.path.exists(latest_path):
        res = {
            "model": model_id,
        }

        with open(latest_path) as f:
            obj = json.load(f)

            for metric in obj["metrics"]:
                if metric != "weight":
                    res["loss"] = obj["metrics"][metric]

            res["epoch"] = obj["epoch"]

        cfg = OmegaConf.load(f"{model_folder}/{model_id}.yaml")

        res["loss name"] = cfg.loss

        if cfg.loss == "L1" and res["loss"] > 0.01 and res["epoch"] > 0:
            res["sino denoiser"] = cfg.parts.sino_denoiser.arch
            res["image denoiser"] = cfg.parts.image_denoiser.arch
            res["deep supervision"] = "side" in cfg.parts
            data.append(res)

df = pd.DataFrame(data)
df = df.sort_values("loss")
print(df)
