import logging
import os
import random
from contextlib import contextmanager
from typing import Union

import numpy as np
import torch

import wandb


@contextmanager
def wandb_disabled():
    """Context manager to temporarily stop wandb from uploading data
    and printing to console."""
    # 1. Save original logging function and terminal silence state
    original_log = wandb.log
    original_silent = os.environ.get("WANDB_SILENT", "false")
    original_disable_level = logging.root.manager.disable

    # 2. Replace wandb.log with a function that does nothing
    wandb.log = lambda *args, **kwargs: None
    os.environ["WANDB_SILENT"] = "true"
    logging.disable(logging.CRITICAL)

    try:
        yield
    finally:
        # 3. Restore everything to normal
        wandb.log = original_log
        os.environ["WANDB_SILENT"] = original_silent
        logging.disable(original_disable_level)


def check_availability(device: Union[str, int]):
    """Checks if the specified device is available and compatible
    with the current environment.

    Parameters
    ----------
    device : Union[str, int]
        The device to verify, which can be a string (e.g., "cpu", "cuda:0")
        or an integer representing the GPU index.
    """
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not available, set device to 'cpu'."


def set_seed(seed: int):
    """Set random seed for reproducibility across random, and torch.

    Parameters
    ----------
    seed : int
        The seed value to set for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
