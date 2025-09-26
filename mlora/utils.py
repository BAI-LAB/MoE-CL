import importlib.metadata
import importlib.util
import logging
import os
import sys
from typing import Optional, Tuple, Union

import torch
from packaging import version


def copy_parameters(source: torch.nn.Module, dest: torch.nn.Module):
    dest.load_state_dict(source.state_dict())
    dest.requires_grad_(False)


def setup_logging(rank: int, world_size: int, log_level: str = "INFO", log_file: str = None):
    """
    Setup logging for distributed training.
    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes.
        log_level (str): Logging level. Defaults to 'INFO'.
        log_file (str): Path to save logs. If None, only prints on stdout.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers
    
    # Set up a custom format
    formatter = logging.Formatter(
        f"[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] [rank {rank}/{world_size}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set up a stream handler for printing to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Set up a file handler for saving logs
    if log_file is not None:
        # Create directory if it doesn't exist
        base, ext = os.path.splitext(log_file)
        rank_log_file = f"{base}_rank{rank}{ext}"
        file_handler = logging.FileHandler(rank_log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def is_package_available(pkg_name: str, pkg_version: Optional[str] = None) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(pkg_version)
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class BitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Quantization not supported.")


class NoneContexts(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
