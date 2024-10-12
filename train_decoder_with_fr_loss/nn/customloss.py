import torch
from torch import nn


def laplacian_norm(tensor):
    f = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]).expand(3, 1, 3, 3).to(tensor.device)
    laplace = torch.nn.functional.conv2d(tensor, f, stride=1, padding=1, groups=3)
    return torch.linalg.norm(laplace, dim=(2, 3)) / (tensor.shape[2] * tensor.shape[3])


class CustomLossForAutoencoder(nn.Module):

    def __init__(self, mse_k=1.0, ll_k=1.0, pl_k=1.0):
        super(CustomLossForAutoencoder, self).__init__()
        self.mse_k = mse_k
        self.ll_k = ll_k
        self.pl_k = pl_k
        self.cos_loss_fn =  torch.nn.CosineEmbeddingLoss()
    def forward(self, _input, target, opf, rpf):
        mse = (_input - target).square().mean()

        ll = 0#(laplacian_norm(_input) - laplacian_norm(target)).square().mean()

        pl = 0
        for key in opf:
            # if key == 'features':
            #     pl += self.cos_loss_fn(opf[key], rpf[key])
            # else:
            pl += (opf[key] - rpf[key]).square().mean()
        pl /= len(opf)

        return mse * self.mse_k + ll * self.ll_k + pl * self.pl_k
