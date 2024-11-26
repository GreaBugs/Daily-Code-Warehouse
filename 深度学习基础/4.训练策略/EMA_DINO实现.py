import torch


class My_model(nn.Module):
    def __init__(self, resnet50):
        super(My_model, self).__init__()
        self.resnet50 = resnet50
        self.mlp = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1024, out_features=256, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=10, bias=True))

    def forward(self, x):
        for name, layer in self.resnet50.named_children():
            if 'fc' not in name:
                x = layer(x)    
        x = x.view(-1, 2048)
        x = self.mlp(x)
        return x


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
    
if __name__=="__main__":
    use_ema = True
    
    resnet50 = models.resnet50(weights="DEFAULT")
    model = My_model(resnet50)
    model.to(device)
    
    # ema
    if use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None