import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_eps = scale_eps

        self.weight = nn.Parameter(torch.randn(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs)) if bias else None

    @property
    def qweight(self):
        scale = self.scale_eps + torch.mean(self.weight.abs()).detach()
        quant = torch.round(self.weight / scale).clamp_(-1, 1)
        return (quant - self.weight).detach() + self.weight

    def forward(self, x):
        return F.linear(x, self.qweight, self.bias)

class BitLinearLora(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.lora_w1 = nn.Parameter(torch.zeros(in_features, in_features//32))
        self.lora_w2 = nn.Parameter(torch.empty(in_features//32, out_features).uniform_(-0.0001, 0.0001))

    def forward(self, x):
        return torch.tanh(x @ self.lora_w1) @ self.lora_w2 + F.linear(x, self.qweight, self.bias)

class BitLinearLoraScaledTwice(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.lora_w1 = nn.Parameter(torch.zeros(in_features, in_features//32))
        self.lora_w2 = nn.Parameter(torch.empty(in_features//32, out_features).uniform_(-0.01, 0.01))
        self.scale_w1 = nn.Parameter(torch.ones(in_features))
        self.scale_w2 = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        return torch.tanh(x @ self.lora_w1) @ self.lora_w2 + self.scale_w2 * F.linear(self.scale_w1 * x, self.qweight, self.bias)

class BitLinearPreAndLoraScaled(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.lora_w1 = nn.Parameter(torch.zeros(in_features, in_features//32))
        self.lora_w2 = nn.Parameter(torch.empty(in_features//32, out_features).uniform_(-0.0001, 0.0001))
        self.scale_w1 = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return (torch.tanh(x @ self.lora_w1) @ self.lora_w2 + 1.0) * F.linear(self.scale_w1 * x, self.qweight, self.bias)

class BitLinearAdditiveScaledTwice(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.scale_w1 = nn.Parameter(torch.ones(in_features))
        self.scale_w2 = nn.Parameter(torch.ones(out_features))
        self.scale_w3 = nn.Parameter(torch.empty(out_features).uniform_(-0.001, 0.001))

    def forward(self, x):
        base = x
        if self.weight.size(0) > x.size(-1):
            base = F.pad(x, [0, self.weight.size(0) - x.size(-1)])
        base = base[..., :self.weight.size(0)]
        return self.scale_w3 * base + self.scale_w2 * F.linear(self.scale_w1 * x, self.qweight, self.bias)

class BitLinearPreScaled(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.scale_w1 = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return F.linear(x, self.scale_w1 * self.qweight, self.bias)

class BitLinearPostScaled(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.scale_w1 = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        return self.scale_w1 * F.linear(x, self.qweight, self.bias)

class BitLinearScaledTwice(BitLinear):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, scale_eps, device, dtype)
        self.scale_w1 = nn.Parameter(torch.ones(in_features))
        self.scale_w2 = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        return self.scale_w2 * F.linear(self.scale_w1 * x, self.qweight, self.bias)

class BitLinearScaledTwiceBias(BitLinearScaledTwice):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        super().__init__(in_features, out_features, True, scale_eps, device, dtype)
