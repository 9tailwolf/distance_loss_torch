import torch
import torch.nn as nn
from torch import Tensor

class DiMSLoss(nn.Module):
    def __init__(self, alpha:int=2) -> None:
        super(DiMSLoss, self).__init__()
        self.alpha = alpha

    def _compute_weights(self, target:Tensor, indices:Tensor) -> Tensor:
        target_index = torch.argmax(target,dim=1).view(-1,1)
        weights = 1+torch.abs(indices - target_index)
        weights = weights ** self.alpha
        return weights

    def _set_dimention_2d(self, tensor:Tensor) -> Tensor:
        if tensor.dim()==1:
            return tensor.unsqueeze(0)
        return tensor
    
    def forward(self, x:Tensor, target:Tensor) -> Tensor:
        x, target = self._set_dimention_2d(x), self._set_dimention_2d(target)
        indices = torch.arange(target.size(1)).unsqueeze(0).repeat(target.size(0),1).to(x.device)
        weights = self._compute_weights(target, indices)
        loss = torch.mean(weights * (x - target)**2, dim=1)
        loss = loss.sum()
        return loss