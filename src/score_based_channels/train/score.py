#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hydra
import numpy as np
import torch, os, copy

from dataclasses import dataclass, field
from tqdm import tqdm as tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses import get_optimizer
from ncsnv2.losses.dsm import anneal_dsm_score_estimation

from score_based_channels.models.loaders import Channels
from torch.utils.data import DataLoader


@dataclass
class ModelConfig:
    ngf: int = 32
    sigma_begin: float = 39.15
    sigma_rate: float = 0.995
    num_classes: int = 2311

    def __post_init__(self):
        # Dynamically calculate sigma_end if not provided
        self.sigma_end = self.sigma_begin * self.sigma_rate ** (self.num_classes - 1)
        # Assign constant values
        self.ema: bool = True
        self.ema_rate: float = 0.999
        self.normalization: str = "InstanceNorm++"
        self.nonlinearity: str = "relu"
        self.sigma_dist: str = "geometric"


@dataclass
class OptimConfig:
    optimizer: str = "Adam"
    lr: float = 0.0001

    def __post_init__(self):
        optimizers = ["Adam", "RMSProp", "SGD"]
        if self.optimizer not in optimizers:
            raise ValueError(f"Invalid optimizer {self.optimizer}! Should be one of {optimizers}")
        # Assign constant values
        self.weight_decay: float = 0.000
        if self.optimizer == "Adam":
            self.beta1: float = 0.9
            self.amsgrad: bool = False
            self.eps: float = 0.001


@dataclass
class TrainingConfig:
    batch_size: int = 32
    n_epochs: int = 400
    num_workers: int = 4

    def __post_init__(self):
        if os.name != "posix" and self.num_workers > 0:
            self.num_workers = 0
            print(
                "Warning: Windows doesn't support num_workers > 0 for PyTorch dataloaders. Using 0 workers!"
            )


@dataclass
class DataConfig:
    channel: str = "CDL-C"
    noise_std: float = 0.0
    image_size: list[int] = field(default_factory=lambda: [16, 64])
    norm_method: str = "global"
    norm_values: list[float] | None = None
    spacing_list: list[float] = field(default_factory=lambda: [0.5])

    logit_transform: bool = False
    rescaled: bool = False

    def __post_init__(self):
        self.channels = len(self.image_size)
        self.num_pilots = self.image_size[1]

        channels = ["CDL-A", "CDL-B", "CDL-C", "CDL-D"]
        if self.channel not in channels:
            raise ValueError(f"Invalid channel {self.channel}! Should be one of {channels}")

        norm_methods = ["global", "entrywise"]
        if self.norm_method not in norm_methods:
            raise ValueError(
                f"Invalid normalization method {self.norm_method}! Should be one of {norm_methods}"
            )

        if self.norm_values and len(self.norm_values) != 2:
            raise ValueError(
                f"Normalization values {self.norm_values} must contain exactly two floats (mean, std)!"
            )


@dataclass
class TrainScoreConfig:
    gpu: int = 0
    train_seed: int = 1234
    val_seed: int = 4321

    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        if self.gpu >= torch.cuda.device_count() or self.gpu < 0:
            raise ValueError(
                f"Invalid GPU {self.gpu} selected! Must be one of {torch.cuda.device_count()} available GPUs"
            )
        self.device = f"cuda:{self.gpu}"


cs = ConfigStore.instance()
cs.store(name="train_score_config", node=TrainScoreConfig)


@hydra.main(version_base=None, config_name="train_score_config")
def main(cfg: TrainScoreConfig):
    config = OmegaConf.to_object(cfg)

    # Environment & backend setup
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    print(f"Starting training on device: {config.device} for channel: {config.data.channel}")

    # Get datasets and loaders for channels
    dataset = Channels(config.train_seed, config, config.data.norm_method, config.data.norm_values)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Validation data
    val_datasets, val_loaders, val_iters = [], [], []
    for idx in range(len(config.data.spacing_list)):
        # Validation config
        val_config = copy.deepcopy(config)
        val_config.data.spacing_list = [config.data.spacing_list[idx]]
        # Create locals
        val_datasets.append(Channels(config.val_seed, val_config, norm_values=[dataset.mean, dataset.std]))
        val_loaders.append(
            DataLoader(
                val_datasets[-1],
                batch_size=len(val_datasets[-1]),
                shuffle=False,
                num_workers=0,
                drop_last=True,
            )
        )
        val_iters.append(iter(val_loaders[-1]))  # For validation

    # Instantiate model and optimizer
    diffuser = NCSNv2Deepest(config).to(config.device)
    optimizer = get_optimizer(config, diffuser.parameters())

    # Instantiate counters and EMA helper
    start_epoch, step = 0, 0
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(diffuser)

    # Get all sigma values for the discretized VE-SDE
    sigmas = get_sigmas(config)

    # Sample fixed validation data
    val_H_list = []
    for idx in range(len(config.data.spacing_list)):
        val_sample = next(val_iters[idx])
        val_H_list.append(val_sample["H_herm"].to(config.device))

    # Logging
    config.log_path = "./models/score/%s" % config.data.channel
    os.makedirs(config.log_path, exist_ok=True)
    train_loss, val_loss = [], []

    # For each epoch
    for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
        # For each batch
        for i, sample in tqdm(enumerate(dataloader)):
            diffuser.train()
            step += 1
            # Move data to device
            for key in sample:
                sample[key] = sample[key].to(config.device)

            # Compute DSM loss using Hermitian channels
            loss = anneal_dsm_score_estimation(diffuser, sample["H_herm"], sigmas, None)

            # Logging
            if step == 1:
                running_loss = loss.item()
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss.item()
            train_loss.append(loss.item())

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            if config.model.ema:
                ema_helper.update(diffuser)

            # Verbose
            if step % 100 == 0:
                if config.model.ema:
                    val_score = ema_helper.ema_copy(diffuser)
                else:
                    val_score = diffuser

                # For each validation setup
                local_val_losses = []
                for idx in range(len(config.data.spacing_list)):
                    with torch.no_grad():
                        val_dsm_loss = anneal_dsm_score_estimation(val_score, val_H_list[idx], sigmas, None)
                    # Store
                    local_val_losses.append(val_dsm_loss.item())
                # Sanity delete
                del val_score
                # Log
                val_loss.append(local_val_losses)

                # Print
                if len(local_val_losses) == 1:
                    print(
                        "Epoch %d, Step %d, Train Loss (EMA) %.3f, Val. Loss %.3f"
                        % (epoch, step, running_loss, local_val_losses[0])
                    )
                elif len(local_val_losses) >= 2:
                    print(
                        "Epoch %d, Step %d, Train Loss (EMA) %.3f, Val. Loss (Split) %.3f %.3f"
                        % (epoch, step, running_loss, local_val_losses[0], local_val_losses[1])
                    )

    # Save final weights
    torch.save(
        {
            "model_state": diffuser.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        os.path.join(config.log_path, "final_model.pt"),
    )


if __name__ == "__main__":
    main()
