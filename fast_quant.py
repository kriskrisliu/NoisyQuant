import os
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from types import MethodType
import torch
from tqdm import tqdm

def quant_activation(x, bit, act_scale):
    n = 2 ** (bit - 1) - 1
    aint = (x / act_scale).round().clamp(-n-1,n)
    x = aint * act_scale
    return x

def quant_forward(self, x: Tensor) -> Tensor:
    # NOTE: the 1st forward should be determine absmax from calibration!!
    if self.act_scale is None:
        n = 2 ** (self.bit - 1) - 1
        self.act_scale = x.data.abs().max() / n

        if self.with_noisy_quant:
            # NoisyQunat implementation
            criterion = torch.nn.MSELoss()
            noisy_bias = (torch.randn_like(x[:1,:1,:])*2-1)*self.act_scale
            loss_min = 1e6
            pbar = tqdm(range(1,100),desc=f"{self.own_name}")
            for ii in pbar:
                candidate = noisy_bias * ii/100
                xq = quant_activation(x + candidate, bit=self.bit, act_scale=self.act_scale)
                xq -= candidate
                zq = F.linear(xq, self.weight, self.bias)
                z = F.linear(x, self.weight, self.bias)
                loss = criterion(zq, z)
                pbar.set_postfix(
                    loss=f"{loss.item():.2e}", 
                    loss_min=f"{loss_min:.2e}",
                    best_candidate_scale=f"{ii/100:.2f}"
                )
                if loss < loss_min:
                    loss_min = loss
                    best_candidate = candidate
            self.noisy_bias = best_candidate
            self.add_noise = True
            x = quant_activation(x + self.noisy_bias, bit=self.bit, act_scale=self.act_scale)
            x -= self.noisy_bias
        else:
            # vanilla quant with no tricks, e.g., clipping, zero-shifting, bias-correction ...
            x = quant_activation(x, bit=self.bit, act_scale=self.act_scale)
    else:
        if self.add_noise:
            x = x + self.noisy_bias
        x = quant_activation(x, bit=self.bit, act_scale=self.act_scale)
        if self.add_noise:
            x = x - self.noisy_bias
    return F.linear(x, self.weight, self.bias)

def fast_quant(model, bit=6, with_noisy_quant=False):
    for name, module in model.named_modules():
        module.own_name = name
        if isinstance(module, nn.Linear) and name!="head":
            module.bit = bit
            n = 2 ** (bit - 1) - 1
            weight = module.weight.data.clone()
            scale_channel_wise = weight.abs().max(dim=1,keepdim=True)[0] / n
            wint = (weight/scale_channel_wise).round().clamp(-n-1,n)
            wq = wint * scale_channel_wise
            module.weight.data = wq.data
            module.act_scale = None
            module.add_noise = False
            module.with_noisy_quant = with_noisy_quant
            module.forward = MethodType(quant_forward, module)

            # import ipdb;ipdb.set_trace()
    return model
