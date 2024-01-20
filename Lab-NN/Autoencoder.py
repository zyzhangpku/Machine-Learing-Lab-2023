import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
H = 24
W = 24


class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * H // 4 * W // 4, encoding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(encoding_dim, 128 * H // 4 * W // 4)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, H // 4, W // 4)
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = torch.sigmoid(self.conv_transpose3(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        self.name = 'AE'
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
