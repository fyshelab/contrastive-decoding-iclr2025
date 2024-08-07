from typing import Sequence, Optional

import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(
            self,
            layer_sizes: Sequence[int],
            dropout_p: Optional[float] = None,
            normalize: bool = False,
            nonlinearity: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        self.normalize = normalize
        self.layers = [nn.Linear(in_features=layer_sizes[0], out_features=layer_sizes[1])]
        for in_size, out_size in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.layers += [
                nonlinearity(),
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_size, out_size)
            ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        self.layers.to(x.device)
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        if self.normalize:
            x = x / torch.norm(x, dim=1, keepdim=True)
        return x
