import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageColumnKLDivLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss between columns of two image tensors.

        :param predicted: Tensor of shape (batch_size, 1, height, width)
        :param target: Tensor of shape (batch_size, 1, height, width)
        :return: scalar loss value
        """
        # ensure inputs are positive
        predicted = torch.clamp(predicted, min=0.0)
        target = torch.clamp(target, min=0.0)

        # add epsilon to inputs before normalization to ensure no zeros
        predicted = predicted + self.epsilon
        target = target + self.epsilon
        
        # ensure inputs are normalized
        pred_norm = predicted / predicted.sum(dim=1, keepdim=True)
        target_norm = target / target.sum(dim=1, keepdim=True)

        
        # calculate KL divergence for each column
        kl_div = F.kl_div(
            torch.log(pred_norm),  # input needs to be log-probabilities
            target_norm,      # target needs to be probabilities
            reduction='none'
        )
        
        # handle nan values
        kl_div = torch.nan_to_num(kl_div, 0.0)

        # sum over height and width dimensions
        loss = kl_div.sum(dim=(1, 2)).mean()  # mean over batch size
        
        return loss