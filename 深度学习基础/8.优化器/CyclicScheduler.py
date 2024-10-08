from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class CyclicScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        interval_steps = [],
        interval_lrs = [],
        last_epoch = -1,
    ):        
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]
    
    def get_lr(self):
        for _i in range(0, len(self.interval_steps)-1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (self.interval_steps[_i + 1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]             
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        #self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: #, self.get_lr()):
            param_group['lr'] = _lr