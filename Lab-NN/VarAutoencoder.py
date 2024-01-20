import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 24, 24


class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * (IMG_WIDTH // 8) * (IMG_HEIGHT // 8), encoding_dim)
        self.fc_log_var = nn.Linear(64 * (IMG_WIDTH // 8) * (IMG_HEIGHT // 8), encoding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarDecoder, self).__init__()

        self.fc = nn.Linear(encoding_dim, 64 * (IMG_WIDTH // 8) * (IMG_HEIGHT // 8))
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, v):
        v = self.fc(v)
        v = v.view(-1, 64, IMG_WIDTH // 8, IMG_HEIGHT // 8)
        v = F.relu(self.deconv1(v))
        v = F.relu(self.deconv2(v))
        x = torch.sigmoid(self.deconv3(v))

        return x


class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def VAE_loss_function(outputs, images):
    reconstructed, mu, log_var = outputs
    recon_loss = F.mse_loss(reconstructed, images, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = recon_loss + kl_div
    return loss
