import torch


class Config:
    num_workers = 0
    batch_size = 4
    n_epochs = 3
    lr = 0.0002

    folder = "effdet5-cutmix-augmix"

    verbose = True
    verbose_step = 1

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    #     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    #     scheduler_params = dict(
    #         max_lr=0.001,
    #         epochs=n_epochs,
    #         steps_per_epoch=int(len(train_dataset) / batch_size),
    #         pct_start=0.1,
    #         anneal_strategy='cos',
    #         final_div_factor=10**5
    #     )

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau

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
