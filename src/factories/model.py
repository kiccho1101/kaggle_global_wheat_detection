import torch
import sys
import gc

from src.config import Config

if Config.model == "timm_effdet":
    sys.path.insert(0, "./input/timm-efficientdet-pytorch")
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
from effdet.efficientdet import HeadNet

from typing import Optional


def get_effdet_train(model_path: Optional[str] = None):
    config = get_efficientdet_config("tf_efficientdet_d5")
    if model_path is None:
        net = EfficientDet(config, pretrained_backbone=True)
    else:
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)
        config.num_classes = 1
        config.image_size = 512
        net.class_net = HeadNet(
            config,
            num_outputs=config.num_classes,
            norm_kwargs=dict(eps=0.001, momentum=0.01),
        )
    return DetBenchTrain(net, config)


def get_effdet_eval(checkpoint_path: str):
    config = get_efficientdet_config("tf_efficientdet_d5")
    model = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = 512

    model.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    gc.collect()

    model = DetBenchEval(model, config)
    model.eval()
    return model
