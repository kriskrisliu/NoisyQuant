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

def quant_weight(w, bit, mode="channel_wise", symmetric=True):
    if mode=="channel_wise" and symmetric:
        n = 2 ** (bit - 1) - 1
        scale_channel_wise = w.abs().max(dim=1,keepdim=True)[0] / n
        wint = (w/scale_channel_wise).round().clamp(-n-1,n)
        wq = wint * scale_channel_wise
    else:
        raise NotImplementedError

    return wq

def percentile_search(x, w, bias, z0, bit, search_space=200):
    """percentile method to determine clipping point

    Args:
        x : raw activation
        w : raw/quanted weight
        bias : raw bias
        z0 : raw x@w
        search_space (int, optional): Defaults to 200.
    """
    absmax = x.abs().max()
    min_loss = None
    best_clip = None
    pbar = tqdm(range(search_space, 0, -1), desc="search clip")
    for ii in pbar:
        clip_value = absmax/search_space*ii
        act_scale = clip_value/(2**(bit-1)-1)
        z = F.linear(quant_activation(x.clamp(-clip_value, clip_value), bit, act_scale), w, bias)
        loss = ((z-z0)**2).mean()
        if min_loss is None:
            min_loss = loss
            best_clip = clip_value
        elif loss < min_loss:
            min_loss = loss
            best_clip = clip_value
            best_act_scale = act_scale
        pbar.set_postfix(
            loss=f"{loss.item():.2e}", 
            loss_min=0 if min_loss is None else f"{min_loss:.2e}",
            absmax=f"{absmax.item():.2e}",
            best_clip=0 if best_clip is None else f"{best_clip.item():.2e}"
        )

    return best_act_scale
        
    
def quant_forward(self, x: Tensor) -> Tensor:
    # NOTE: the 1st forward should be determine absmax from calibration!!
    if self.clip_search:
        z0 = F.linear(x, self.weight, self.bias)
        self.act_scale = percentile_search(x, self.weight, self.bias, z0, self.bit, search_space=1000)
        self.clip_search = False
        xq = quant_activation(x, self.bit, self.act_scale)
        return F.linear(xq, self.weight, self.bias)
    elif self.noisy_search: 
        if self.with_noisy_quant:
            # NoisyQunat implementation
            criterion = torch.nn.MSELoss()
            noisy_bias = (torch.randn_like(x[:1,:1,:])*2-1)*self.act_scale
            search_space_mean = 200
            search_space_range = 1000
            
            if self.search_mean:
                # determine mean of noisy bias
                loss_min = 1e6
                best_candidate = torch.tensor([0.0])
                pbar = tqdm(range(-search_space_mean,search_space_mean),desc=f"noisy mean: {self.own_name}")
                for ii in pbar:
                    candidate = self.act_scale * ii/search_space_mean
                    xq = quant_activation(x + candidate, bit=self.bit, act_scale=self.act_scale)
                    xq -= candidate
                    zq = F.linear(xq, self.weight, self.bias)
                    z = F.linear(x, self.weight, self.bias)
                    loss = criterion(zq, z)
                    pbar.set_postfix(
                        loss=f"{loss.item():.2e}", 
                        loss_min=f"{loss_min:.2e}",
                        best_mean=f"{best_candidate.item():.2e}"
                    )
                    if loss < loss_min:
                        loss_min = loss
                        best_candidate = candidate
                self.noisy_bias = best_candidate
                best_noisy_mean = best_candidate
            else:
                self.noisy_bias = 0.
                best_noisy_mean = 0.
            
            if self.search_noisy:
                # determine range of noisy bias
                loss_min = 1e6
                best_candidate_scale = 0
                pbar = tqdm(range(0,search_space_range*2),desc=f"noisy range: {self.own_name}")
                for ii in pbar:
                    candidate = best_noisy_mean + noisy_bias * ii/search_space_range
                    xq = quant_activation(x + candidate, bit=self.bit, act_scale=self.act_scale)
                    xq -= candidate
                    zq = F.linear(xq, self.weight, self.bias)
                    z = F.linear(x, self.weight, self.bias)
                    loss = criterion(zq, z)
                    pbar.set_postfix(
                        loss=f"{loss.item():.2e}", 
                        loss_min=f"{loss_min:.2e}",
                        best_range=f"{best_candidate_scale:.2e}"
                    )
                    if loss < loss_min:
                        loss_min = loss
                        best_candidate = candidate
                        best_candidate_scale = ii/search_space_range
                self.noisy_bias = best_candidate
            else:
                self.noisy_bias = best_noisy_mean
            self.add_noise = True
            self.noisy_search = False
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

def fast_quant(model, bit=6, with_noisy_quant=False, percentile=False, search_mean=False, search_noisy=False):
    for name, module in tqdm(model.named_modules(), desc="Quantize weights"):
        module.own_name = name
        if isinstance(module, nn.Linear) and name!="head":
            module.bit = bit
            w = module.weight.data.clone()
            wq = quant_weight(w, bit, mode="channel_wise", symmetric=True)
            module.original_weight = w
            module.weight.data = wq.data
            module.act_scale = None
            module.add_noise = False
            module.with_noisy_quant = with_noisy_quant
            if with_noisy_quant:
                module.clip_search = percentile
                module.noisy_search = (search_mean or search_noisy)
                module.search_mean = search_mean
                module.search_noisy = search_noisy                
            else:
                module.clip_search = False
                module.noisy_search = False
            module.forward = MethodType(quant_forward, module)

    return model
