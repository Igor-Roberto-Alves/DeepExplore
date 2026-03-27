import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class sGuidedVAE(nn.Module):

    def __init__(self, latent_dim=8):

        super().__init__()

        self.latent_dim = latent_dim

        # Encoder

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True)
        )

        self.fc = nn.Linear(256 * 8 * 8, latent_dim * 2)

        # Decoder

        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )


        # Classificador principal

        self.classifier = nn.Sequential(

            nn.Linear(1, 32),
            nn.ReLU(True),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Classificador adversarial

        self.adv_classifier = nn.Sequential(

            nn.Linear(latent_dim - 1, 32),
            nn.ReLU(True),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(x.size(0), -1)

        h = self.fc(h)

        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]

        return mu, logvar


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std



    def decode(self, z):

        h = self.fc_dec(z)

        h = h.view(z.size(0), 256, 8, 8)

        return self.decoder(h)


    # Forward

    def forward(self, x, lambda_adv=1.0):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        cls_out = self.classifier(mu[:, 0:1])

        z_adv = grad_reverse(mu[:, 1:], lambda_adv)

        adv_out = self.adv_classifier(z_adv)

        return recon, mu, logvar, cls_out, adv_out