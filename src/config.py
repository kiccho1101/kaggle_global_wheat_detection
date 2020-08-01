import torch
import mlflow


class Config:
    seed = 42
    device = torch.device("cuda")
    num_workers = 0
    batch_size = 2
    n_folds = 5
    n_epochs = 3
    n_epochs_after_pl = 3
    lr = 0.0002
    exp_name: str = "cv"
    logdir = "./logs"
    pseudo_labeling: bool = True
    pseudo_labeling_threshold: float = 0.05

    model: str = "timm_effdet"
    folder = "effdet5-cutmix-augmix"

    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
    )

    def __init__(self, WORK_DIR: str = "."):
        self.WORK_DIR = WORK_DIR
        self.INPUT_DIR = f"{WORK_DIR}/input/global-wheat-detection"
        self.effdet_path = f"{WORK_DIR}/input/efficientdet/efficientdet_d5-ef44aea8.pth"

    def log_mlflow_params(self):
        mlflow.log_param("num_workers", self.num_workers)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("model_name", self.model)
        mlflow.log_param("effdet_path", self.effdet_path)
        mlflow.log_param("n_folds", self.n_folds)
        mlflow.log_param("n_epochs", self.n_epochs)
        mlflow.log_param("n_epochs_after_pl", self.n_epochs_after_pl)
        mlflow.log_param("pseudo_labeling", self.pseudo_labeling)
        mlflow.log_param("pseudo_labeling_threshold", self.pseudo_labeling_threshold)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("folder", self.folder)
        mlflow.log_param("verbose", self.verbose)
        mlflow.log_param("verbose_step", self.verbose_step)
        mlflow.log_param("step_scheduler", self.step_scheduler)
        mlflow.log_param("validation_scheduler", self.validation_scheduler)
        mlflow.log_param("scheduler_class", self.scheduler_class.__name__)
        mlflow.log_param("verbose_step", self.verbose_step)
        mlflow.log_params(
            {f"scheduler_params_{k}": v for k, v in self.scheduler_params.items()}
        )
