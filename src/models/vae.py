import torch
import torch.nn as nn

from .blocks import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, is_mse_loss=True, device="cpu"):
        super().__init__()

        self.is_mse_loss = is_mse_loss
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim, is_mse_loss)

        self.to(device)
        self.z_dim = z_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        mean, logvar = self.encoder.forward(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder.forward(z)
        return x_recon, mean, logvar
