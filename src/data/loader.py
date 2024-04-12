#!/usr/bin/env python3

"""Data loader."""
import torch
import scipy.io
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Subset
import torchvision.datasets as tvd
from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset, CiFar100Dataset
)

logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
    'cifar100': CiFar100Dataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        if cfg.DATA_TRAIN_TYPE:
            dataset = _DATASET_CATALOG[dataset_name](cfg, split)

        else:
            if split == 'train':
                dataset = tvd.CIFAR100(root='Datasets\\CIFAR-100', download=False, transform=transform, train=True)
                # subset_size = int(0.6 * len(datas))
                # print("Train Size = ", subset_size)
                # # Generate random indices for the subset
                # indices = torch.randperm(len(datas)).tolist()

                # # Create the subset
                # dataset = Subset(datas, indices[:subset_size])
                
            else:
                dataset = tvd.CIFAR100(root='Datasets\\CIFAR-100', download=False, transform=transform, train=False)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
