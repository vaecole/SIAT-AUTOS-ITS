
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(

        )

        self.decoder = nn.Sequential(

        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x