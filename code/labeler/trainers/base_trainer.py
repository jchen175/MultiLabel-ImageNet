import torch
import random
import numpy as np

def get_optimizer(models, args):
    if type(models) == list:
        params = []
        for model in models:
            params += list(model.parameters())
    else:
        params = models.parameters()
    if args.optimizer == 'sgd':
        weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer


def get_scheduler(optimizer, args):
    # update learning rate every epoch
    if args.scheduler is None or args.scheduler.lower() == 'none':
        # dummy scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif args.scheduler == 'warmup_cosine':
        warmup_epochs = args.warmup_epochs
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=args.epochs - warmup_epochs, eta_min=0.0)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                             schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                             milestones=[warmup_epochs])
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    return lr_scheduler

def get_stepwise_scheduler(optimizer, args):
    # update learning rate every iteration
    if args.scheduler is None:
        # dummy scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif args.scheduler == 'warmup_cosine':
        total_iters = args.epochs * args.iters_per_epoch
        warmup_iters = args.warmup_epochs * args.iters_per_epoch

        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max= total_iters - warmup_iters, eta_min=0.0)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                                milestones=[warmup_iters])
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    return lr_scheduler

def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Set seed for reproducibility.

    Args:
        seed (int): Seed value.
        deterministic (bool): If True, sets deterministic flags in PyTorch.
    """
    torch.backends.cudnn.benchmark = True if not deterministic else False
    torch.backends.cudnn.deterministic = deterministic
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EarlyStoppingDDP:
    """
    Stop training when the monitored metric has not improved for <patience> epochs.
    Works in DDP: the decision is made on rank-0 and broadcast to all ranks.
    """
    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode  # "min" for loss, "max" for accuracy
        self.best       = None
        self.num_bad    = 0
        self.should_stop_flag = False

    def _is_better(self, metric):
        if self.best is None:
            return True
        if self.mode == "min":
            return metric < self.best - self.min_delta
        else:  # "max"
            return metric > self.best + self.min_delta

    def step(self, metric):
        if self._is_better(metric):
            self.best    = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop_flag = True

    # convenient property
    @property
    def should_stop(self):
        return self.should_stop_flag
