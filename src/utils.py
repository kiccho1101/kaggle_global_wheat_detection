import pandas as pd
import numpy as np
import time
from typing import List, Any
from contextlib import contextmanager
import os
import glob
import random
import torch
import mlflow

from typing import Tuple

from src.config import Config


@contextmanager
def timer(name: str, mlflow_on: bool = False):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.4f} s")
    print()
    if mlflow_on:
        mlflow.log_param(name, f"{time.time() - t0:.4f}s")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def start_mlflow(config: Config) -> Tuple[str, str]:
    try:
        mlflow.end_run()
    except Exception:
        pass
    if mlflow.get_experiment_by_name(config.exp_name) is None:
        mlflow.create_experiment(config.exp_name)
    experiment_id: str = mlflow.get_experiment_by_name(config.exp_name).experiment_id
    print("put the run name")
    run_name: str = input()
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    config.log_mlflow_params()
    return experiment_id, run_name


def remove_empty_dirs(path: str, remain: int = 3):
    for delete_path in sorted(
        [
            dir_path
            for dir_path in glob.glob(f"{path}/*")
            if len(glob.glob(f"{dir_path}/*")) == 0
        ]
    )[:-remain]:
        os.rmdir(delete_path)
