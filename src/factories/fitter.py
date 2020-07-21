import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import os
import glob
from tqdm.autonotebook import tqdm
import numpy as np
import numba

from src.config import Config
from src.utils import timer
from src.types import Imgs, Boxes, Labels
from src.factories.model import get_effdet_train, get_effdet_eval
from src.factories.loss_fn import get_average_meter
from src.factories.tta import get_tta_transforms, make_tta_predictions
from src.factories.wbf import run_wbf
from src.factories.metric import calculate_image_precision

from typing import Optional, List
import mlflow
import mlflow.pytorch


class Fitter:
    def __init__(
        self, cv_num: int, config: Config,
    ):
        self.config: Config = config
        self.epoch: int = 0
        self.cv_num: int = cv_num
        self.start_time: str = datetime.datetime.now().isoformat()
        self.log_path: str = f"{self.config.WORK_DIR}/output/{self.start_time}"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        self.best_summary_loss: float = 10 ** 5

        self.train_model = self.load_train_model()
        self.loss_fn = get_average_meter()
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.train_model.parameters(), lr=config.lr
        )
        self.scheduler = config.scheduler_class(
            self.optimizer, **config.scheduler_params
        )

    def load_train_model(self):
        if self.config.model == "effdet" or self.config.model == "timm_effdet":
            model = get_effdet_train(self.config.effdet_path)
            model.to(self.config.device)
        else:
            model = get_effdet_train(self.config.effdet_path)
            model.to(self.config.device)
        return model

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        with_validation: bool = False,
    ) -> None:
        for _ in range(self.config.n_epochs):

            with timer(f"CV {self.cv_num} epoch {self.epoch}", mlflow_on=True):
                summary_loss = self._train_one_epoch(train_loader)
                self.save(f"{self.log_path}/last-checkpoint_cv{self.cv_num}.bin")

                if with_validation:
                    summary_loss = self._validation(valid_loader)

                if summary_loss.avg < self.best_summary_loss:
                    self.best_summary_loss = summary_loss.avg
                    self.train_model.eval()
                    self.save(f"{self.log_path}/best-checkpoint_cv{self.cv_num}.bin")

                if self.config.validation_scheduler:
                    self.scheduler.step(metrics=summary_loss.avg)

                self.epoch += 1

    def predict_and_evaluate(
        self, valid_loader: DataLoader, checkpoint_path: Optional[str] = None
    ) -> float:
        if checkpoint_path is None:
            checkpoint_path = f"{self.log_path}/best-checkpoint_cv{self.cv_num}.bin"

        precisions: List[float] = []
        tta_transforms = get_tta_transforms()
        model = get_effdet_eval(checkpoint_path).to(self.config.device)
        for step, (images, targets, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            predictions = make_tta_predictions(
                model, images, tta_transforms, 0.25, self.config.device
            )
            image_precisions: List[float] = []
            for image_index, image in enumerate(images):
                gts: Boxes = targets[image_index]["bboxes"].cpu().numpy()
                preds, scores, labels = run_wbf(
                    predictions,
                    image_index,
                    image_size=512,
                    iou_thr=0.44,
                    skip_box_thr=0.43,
                    weights=None,
                )
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted: Boxes = preds[preds_sorted_idx]
                iou_thresholds = numba.typed.List()
                for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                    iou_thresholds.append(x)
                image_precision = calculate_image_precision(
                    gts, preds_sorted, thresholds=iou_thresholds, form="pascal_voc"
                )
                image_precisions.append(image_precision)
            precision = np.mean(image_precisions)
            precisions.append(precision)
            print(
                f"CV {self.cv_num} Eval Step {step}/{len(valid_loader)}, "
                + f"precision: {precision}, "
            )
        avg_precision = np.mean(precisions)
        mlflow.log_metric(f"precision_cv_{self.cv_num}", avg_precision)
        return avg_precision

    def _train_one_epoch(self, train_loader: DataLoader):
        self.train_model.train()
        summary_loss = self.loss_fn

        start = time.time()

        for step, (images, targets, _) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            if self.config.verbose:
                print(
                    f"CV {self.cv_num} EPOCH {self.epoch} Train Step {step}/{len(train_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, "
                    + f"time: {(time.time() - start):.5f}",
                )
            images = torch.stack(images)
            images: Imgs = images.to(self.config.device).float()
            bboxes: Boxes = [
                target["bboxes"].to(self.config.device).float() for target in targets
            ]
            labels: Labels = [
                target["labels"].to(self.config.device).float() for target in targets
            ]

            target_res = {"bbox": bboxes, "cls": labels}

            self.optimizer.zero_grad()

            if self.config.model == "effdet":
                outputs = self.train_model(images, target_res)
                loss = outputs["loss"]
            elif self.config.model == "timm_effdet":
                loss, _, _ = self.train_model(images, bboxes, labels)
            else:
                loss, _, _ = self.train_model(images, bboxes, labels)

            loss.backward()
            summary_loss.update(loss.detach().item(), self.config.batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            mlflow.log_metric(
                f"train_loss_cv_{self.cv_num}",
                summary_loss.avg,
                step=self.epoch * len(train_loader) + step,
            )

        return summary_loss

    def _validation(self, valid_loader: DataLoader):
        # TODO: Find a way to execute this with eval mode
        if self.config.model == "effdet":
            self.train_model.train()
        else:
            self.train_model.eval()
        summary_loss = self.loss_fn
        start = time.time()
        for step, (images, targets, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f"CV {self.cv_num} EPOCH {self.epoch} Val Step {step}/{len(valid_loader)}, "
                        + f"summary_loss: {summary_loss.avg:.5f}, "
                        + f"time: {(time.time() - start):.5f}",
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images: Imgs = images.to(self.config.device).float()
                bboxes: Boxes = [
                    target["bboxes"].to(self.config.device).float()
                    for target in targets
                ]
                labels: Labels = [
                    target["labels"].to(self.config.device).float()
                    for target in targets
                ]

                target_res = {
                    "bbox": bboxes,
                    "cls": labels,
                }

                if self.config.model == "effdet":
                    outputs = self.train_model(images, target_res)
                    loss = outputs["loss"]
                elif self.config.model == "timm_effdet":
                    loss, _, _ = self.train_model(images, bboxes, labels)
                else:
                    loss, _, _ = self.train_model(images, bboxes, labels)

                summary_loss.update(loss.detach().item(), batch_size)

            mlflow.log_metric(
                f"valid_loss_cv_{self.cv_num}",
                summary_loss.avg,
                step=self.epoch * len(valid_loader) + step,
            )

        return summary_loss

    def save(self, path: str):
        self.train_model.eval()
        torch.save(
            {
                "model_state_dict": self.train_model.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )


def get_fitter(cv_num: int, config: Config) -> Fitter:
    return Fitter(cv_num, config)
