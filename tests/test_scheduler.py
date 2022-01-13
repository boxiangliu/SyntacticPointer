from torch import optim
from lightning.optim import ExponentialScheduler
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR


def get_optimizer():
    model = nn.Linear(10, 10)
    init_lr = 0.001
    optimizer = SGD(model.parameters(), lr=init_lr)
    return optimizer, init_lr


def test_ExponentialScheduler():
    optimizer, init_lr = get_optimizer()
    gamma = 0.01
    scheduler = ExponentialScheduler(
        optimizer, gamma=gamma, warmup_steps=0, init_lr=init_lr
    )

    assert optimizer.param_groups[0]["lr"] == init_lr
    scheduler.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == init_lr * gamma


def test_ExponentialLR():
    optimizer, init_lr = get_optimizer()
    gamma = 0.01
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    assert optimizer.param_groups[0]["lr"] == init_lr
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == init_lr * gamma
