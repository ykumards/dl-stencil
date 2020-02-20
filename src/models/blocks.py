"""
Contains module containing building blocks of the network.
In this case, decoder and encoder
"""
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, is_mse_loss=True):
        super().__init__()
        self.is_mse_loss = is_mse_loss

        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2,
                padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2,
                padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        if is_mse_loss:
            self.conv8 = nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv8 = nn.ConvTranspose2d(
                16, 3 * 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2img = nn.Linear(hidden_dim, 784)

        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc_bn3(self.fc3(z)))
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = x.view(-1, 16, 8, 8)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))

        if self.is_mse_loss:
            return x.view(-1, 3, 32, 32)
        else:
            return x.view(-1, 256, 3, 32, 32)  # for ce_loss 256 categories


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2mean = nn.Linear(512, 512)
        self.fc2scale = nn.Linear(512, 512)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 8 * 8 * 16)
        x = self.relu(self.fc_bn1(self.fc1(x)))
        mean = self.fc2mean(x)
        logvar = self.fc2scale(x)
        return mean, logvar
