from torch.optim.optimizer import Optimizer


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:  # param_group is a list of dictionaries
                group.setdefault("initial_lr", group["lr"])
                last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer."format(i))
        self.base_lrs = list(map(lambda group: group["initial_lr"], optimizer.param_groups))

    def state_dict(self):
        """Returns the state of the scheduler as a class: `dict`.

        It contains an entry for every variable in self.__dict__ which is not the optimizer. 
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """
        Loads the schedulers state.
        
        Args:
            state_dict: dictionary
                scheduler state. Should be an object returned from a call to :meth: `state_dict`.

        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_groups["lr"] = lr

    def reset_state(self):
        self.optimizer.state.clear()
