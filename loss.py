from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

def off_diagonal(x):

    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class TwinsLoss(torch.nn.Module):

    def __init__(self,device, lambda_param=5e-5):
        super(TwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):

        N = z_a.size(0)
        D = z_a.size(1)


        c = torch.mm(z_a.T, z_b) /N

        c_diff = (c - torch.eye(D,device=self.device)).pow(2)

        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()

    return loss
class SCEMSELoss():
    def __init__(self,device):

       self.device = device

    def cal(self,  x_bar,graph, gamma ):
        loss = gamma * sce_loss(x_bar, graph.ndata["feat"]) + (1-gamma)* F.mse_loss(x_bar, graph.ndata["feat"])

        return loss

