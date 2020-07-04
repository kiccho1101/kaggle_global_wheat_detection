import pandas as pd
import numpy as np
import time
from typing import List, Any
from contextlib import contextmanager
import os
import random
import torch
import mlflow


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
