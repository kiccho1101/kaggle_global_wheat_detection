import torch


class Config:
    num_workers = 0
    batch_size = 1
    n_epochs = 1
    lr = 0.0002

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
