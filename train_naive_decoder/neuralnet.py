import torch
import torch.nn as nn
from typing import List


class ConvTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvTBlock, self).__init__()
        # sequential block consisting of a transposed 2d convolution,
        # batch normalization, and leaky relu activation
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,  # number of input channels
                out_channels,  # number of output channels
                kernel_size=3,  # size of the convolutional kernel
                stride=2,  # stride of the convolution
                padding=1,  # padding added to the input
                output_padding=1,  # additional padding added to the output
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ConvFaceDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super(ConvFaceDecoder, self).__init__()
        hidden_dims = [128, 192, 256, 320, 384, 512, 512]
        hidden_dims.reverse()
        self.decoder = nn.Sequential(
            *[ConvTBlock(in_f, out_f) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])]
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=5,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.PReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=0),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    latent_dim = 512
    dummy = torch.randn(size=(10, latent_dim))
    model = ConvFaceDecoder(latent_dim=latent_dim)
    out = model(dummy)
    print(dummy.shape, out.shape)



