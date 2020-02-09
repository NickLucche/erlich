import json
import os
import sys
import time
from glob import glob
import torch
import torch.jit
import torch.nn as nn
from omegaconf import OmegaConf

from .saver import ModelSaver
from .logging import TrainLogger
from .trainer import BaseTrainer

USAGE = """usage"""


class Erlich:
    def __init__(self, config_folder, model_folder, part_constructor):
        self.config_folder = config_folder
        self.model_folder = model_folder
        self.part_constructor = part_constructor

    def config_from_cli(self):
        if len(sys.argv) < 2:
            print(USAGE)
            sys.exit(1)
        return self.parse_config(sys.argv[1], sys.argv[2:])

    def parse_config(self, name: str, additional: list):
        if name.endswith(".yaml"):
            name = name[:-5]

        # read configuration
        path = os.path.join(self.config_folder, name + ".yaml")
        conf = OmegaConf.load(path)

        # merge with additional settings
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(additional))

        # load requested configs, merge, then remove load list
        if "load" in conf:
            loads = []
            for name in conf.get("load", []):
                loads.append(OmegaConf.load(os.path.join(self.config_folder, name + ".yaml")))
            conf = OmegaConf.merge(*loads, conf)
            conf.pop("load")

        # for each part load configuration if is requested
        for part_name in conf.parts:
            if "load" in conf.parts[part_name]:
                load_path = conf.parts[part_name]["load"]
                if "@" in load_path:
                    load_path, checkpoint = load_path.split("@")
                else:
                    checkpoint = "latest"

                model_name, src_part_name = conf.parts[part_name]["load"].split(".")
                src_model_cfg = OmegaConf.load(os.path.join(self.model_folder, model_name + ".yaml"))
                part_cfg = src_model_cfg.parts[src_part_name]
                conf.parts[part_name] = OmegaConf.merge(part_cfg, conf.parts[part_name])
                conf.parts[part_name].pop("load")

                # if weights are not specified load also weights
                if "weights" not in conf.parts[part_name]:
                    conf.parts[part_name]["weights"] = f"{model_name}.{src_part_name}@{checkpoint}"

        # Fill mandatory parameters with defaults
        conf["batch_size"] = conf.get("batch_size", 16)
        conf["epochs"] = conf.get("epochs", 10)
        conf["validation_batch_size"] = conf.get("validation_batch_size", conf.batch_size * 2)

        return conf

    def instantiate_model_parts(self, cfg: OmegaConf, device):
        parts_cfg = cfg.parts

        parts = dict()

        for name in parts_cfg:
            print(f"Instantiating model part '{name}'")
            part = parts_cfg[name]
            assert "arch" in part or "architecture" in part
            arch = part.get("arch", part.get("architecture", None))

            parts[name] = self.part_constructor(arch, part, cfg).to(device)

            if "weights" in part:
                print(f"    Loading weights from {part.weights}")
                # TODO

            if "jit" in part and part["jit"]:
                try:
                    # TODO proper exception if key is not found
                    shape = [x if isinstance(x, int) else int(cfg.select(x)) for x in part["jit"]]
                except Exception as e:
                    print(f"ERROR in parsing JIT shape for '{name}', skipping JIT\n", e)
                    continue
                print(f"    JIT tracing with input shape = {shape}")
                parts[name] = torch.jit.trace_module(parts[name],
                                                     {"forward": torch.zeros(*shape).to(device)})

        return parts

    def get_next_id(self):
        return str(len(glob(os.path.join(self.model_folder, "*.yaml"))))

    def create_model(self, trainer_class, cfg, device) -> BaseTrainer:
        print("="*20, "MODEL CONFIG", "="*20)
        print(cfg.pretty())
        print("=" * 20, "INSTANTIATING MODEL FOR TRAINING", "=" * 20)

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        # get model ID and path
        mdl_id = self.get_next_id()
        mdl_path = os.path.join(self.model_folder, mdl_id)
        cfg_path = f"{mdl_path}.yaml"

        # instantiate logger and saver
        logger = TrainLogger(mdl_path + ".log", cfg.epochs)
        saver = ModelSaver(mdl_path)

        model_parts = self.instantiate_model_parts(cfg, device)

        # create model trainer
        trainer = trainer_class(cfg, model_parts, saver, logger, device)
        assert isinstance(trainer, BaseTrainer)
        trainer.create_dataloaders()

        # instantiate optimizers
        print("Instantiating optimizers")
        trainer.instantiate_optimizers(cfg)

        # load checkpoint
        # TODO move to trainer
        if "load_checkpoint" in cfg:
            if "@" in cfg["load_checkpoint"]:
                load_id, load_batch = cfg["load_checkpoint"].split("@")
            else:
                load_id, load_batch = cfg["load_checkpoint"], "latest"

            checkpoint = torch.load(os.path.join(self.model_folder, load_id, load_batch+".pth"), map_location=device)
            for k in checkpoint["parts"]:
                trainer.model_parts[k].load_state_dict(checkpoint["parts"][k])
            for k in checkpoint["optimizers"]:
                trainer.optimizers[k].load_state_dict(checkpoint["optimizers"][k])
            # TODO amp loading should be done after initialization
            # if "amp" in checkpoint and checkpoint["amp"] is not None:
            #     amp.load_state_dict(checkpoint["amp"])

        # save config
        OmegaConf.save(cfg, cfg_path)

        return trainer
        #
        # mdl_path = os.path.join(self.folder, str(mdl_id))
        # mdl = self.model_constructor(arch, params)
        # logger = TrainLogger(mdl_path + ".log", trainer.epochs)
        # saver = ModelSaver(self, mdl_id, arch, params, trainer.batch_size, trainer.optimizers_cfg, trainer.epochs,
        #                    kwargs)
        #
        # trainer.setup(mdl, logger, saver)
        #
        # return mdl


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
            raise Exception(
                "model_constructor is None\nPlease specify model_constructor function in ModelManager constructor")

        mdl_id = self.get_next_id()
        mdl_path = os.path.join(self.folder, str(mdl_id))
        mdl = self.model_constructor(arch, params)
        logger = TrainLogger(mdl_path + ".log", trainer.epochs)
        saver = ModelSaver(self, mdl_id, arch, params, trainer.batch_size, trainer.optimizers_cfg, trainer.epochs,
                           kwargs)

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
