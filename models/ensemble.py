import torch

def attempt_load(weights, map_location=None):
    ckpt = torch.load(weights, map_location=map_location)  # load
    return ckpt['model'].float().eval()
