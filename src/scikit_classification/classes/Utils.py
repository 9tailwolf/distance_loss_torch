import torch
import json
import numpy as np

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def str2loss(v):
    if v.lower() in ('mse','ms','m'):
        return 'MSE'
    elif v.lower() in ('ce','crossentropy','c'):
        return 'CE'
    elif v.lower() in ('dims','d'):
        return 'DiMS'
    elif v.lower() in ('oll'):
        return 'OLL'

    else:
        raise 'Loss Error'
