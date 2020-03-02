import json
import os
import sys
import time
from glob import glob
import torch
import torch.jit
import torch.nn as nn
from omegaconf import OmegaConf

import torch.multiprocessing as mp
import torch.distributed as dist


from .saver import ModelSaver
from .logging import TrainLogger
from .trainer import BaseTrainer

USAGE = """usage"""


def run_train(rank, this, *args):
    # print(rank, this, args)
    this._train(rank, *args)


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

    def read_model_config(self, model_id):
        path = os.path.join(self.model_folder, model_id + ".yaml")
        return OmegaConf.load(path)

    def instantiate_model_parts(self, cfg: OmegaConf, device, jit=True):
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

            if "jit" in part and part["jit"] and jit:
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

    def get_checkpoint(self, checkpoint_name):
        """
        Parse a checkpoint name and convert to path
        Examples:
        "0" --> (model_id=0, batch="latest", ...)
        "1@0.100" --> (model_id=1, batch="0.100", ...)
        :param checkpoint_name: Name of the checkpoint
        :return: ID of the model, batch, path of the checkpoint
        """

        if "@" in checkpoint_name:
            model_id, load_batch = checkpoint_name.split("@")
        else:
            model_id, load_batch = checkpoint_name, "latest"

        checkpoint_path = os.path.join(self.model_folder, model_id, load_batch + ".pth")

        return model_id, load_batch, checkpoint_path

    @staticmethod
    def load_state_dicts(model_parts, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for k in checkpoint["parts"]:
            model_parts[k].load_state_dict(checkpoint["parts"][k])

        return checkpoint

    def load_model(self, checkpoint_name, device, jit=False):
        model_id, _, checkpoint_path = self.get_checkpoint(checkpoint_name)
        cfg = self.read_model_config(model_id)

        model_parts = self.instantiate_model_parts(cfg, device, jit=jit)
        self.load_state_dicts(model_parts, checkpoint_path, device)
        return model_parts, cfg

    def _train(self, rank, world_size, devices, trainer_class, cfg, mdl_id, mdl_path, validate_every, logger_min_wait):
        # initialize the process group
        dist.init_process_group("nccl", init_method='file:///code/sharedfile', rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)

        device = torch.device("cuda", devices[rank])
        print(f"Spawned trainer process {rank} that will use GPU device {device}")

        if rank == 0:
            print("=" * 20, "INSTANTIATING MODEL FOR TRAINING", "=" * 20)

            # instantiate logger and saver
            logger = TrainLogger(mdl_path + ".log", cfg.epochs)
            saver = ModelSaver(mdl_path)
        else:
            logger = None
            saver = None

        model_parts = self.instantiate_model_parts(cfg, device)

        # create model trainer
        trainer = trainer_class(cfg, model_parts, saver, logger, device, rank, world_size)
        assert isinstance(trainer, BaseTrainer)
        trainer.create_dataloaders()

        # instantiate optimizers
        if rank == 0:
            print("Instantiating optimizers")
        trainer.instantiate_optimizers(cfg)

        # load checkpoint
        if "load_checkpoint" in cfg:
            _, _, checkpoint_path = self.get_checkpoint(str(cfg["load_checkpoint"]))
            checkpoint = self.load_state_dicts(trainer.model_parts, checkpoint_path, device)

            if "load_optimizers" not in cfg or cfg["load_optimizers"]:
                for k in checkpoint["optimizers"]:
                    trainer.optimizers[k].load_state_dict(checkpoint["optimizers"][k])
            # TODO amp loading should be done after initialization
            # if "amp" in checkpoint and checkpoint["amp"] is not None:
            #     amp.load_state_dict(checkpoint["amp"])

        trainer.train(validate_every, logger_min_wait, distributed_data_parallel=world_size > 1)

    def train(self, trainer_class, cfg, devices, validate_every=-1, logger_min_wait=5):
        print("=" * 20, "MODEL CONFIG", "=" * 20)
        print(cfg.pretty())

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        # get model ID and path
        mdl_id = self.get_next_id()
        mdl_path = os.path.join(self.model_folder, mdl_id)
        cfg_path = f"{mdl_path}.yaml"

        # save config
        OmegaConf.save(cfg, cfg_path)

        world_size = len(devices)
        mp.spawn(run_train,
                 args=(self, world_size, devices, trainer_class, cfg, mdl_id, mdl_path, validate_every, logger_min_wait),
                 nprocs=world_size,
                 join=True)

        #
        # print("=" * 20, "INSTANTIATING MODEL FOR TRAINING", "=" * 20)
        #
        # # instantiate logger and saver
        # logger = TrainLogger(mdl_path + ".log", cfg.epochs)
        # saver = ModelSaver(mdl_path)
        #
        # model_parts = self.instantiate_model_parts(cfg, device)
        #
        # # create model trainer
        # trainer = trainer_class(cfg, model_parts, saver, logger, device)
        # assert isinstance(trainer, BaseTrainer)
        # trainer.create_dataloaders()
        #
        # # instantiate optimizers
        # print("Instantiating optimizers")
        # trainer.instantiate_optimizers(cfg)
        #
        # # load checkpoint
        # if "load_checkpoint" in cfg:
        #     _, _, checkpoint_path = self.get_checkpoint(str(cfg["load_checkpoint"]))
        #     checkpoint = self.load_state_dicts(trainer.model_parts, checkpoint_path, device)
        #
        #     if "load_optimizers" not in cfg or cfg["load_optimizers"]:
        #         for k in checkpoint["optimizers"]:
        #             trainer.optimizers[k].load_state_dict(checkpoint["optimizers"][k])
        #     # TODO amp loading should be done after initialization
        #     # if "amp" in checkpoint and checkpoint["amp"] is not None:
        #     #     amp.load_state_dict(checkpoint["amp"])
        #
        # return trainer
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
