import pytorch_lightning as pl
import torch
from cleanfid import fid
from torch.nn import BCELoss
from torch.optim import Adam

from geneve.models import Discriminator, LatentGenerator

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class LatentGANModule(pl.LightningModule):
    def __init__(self, generator, discriminator, lr: float = 3e-4):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.adversarial_criterion = BCELoss()
        self.lr = lr

    def latent(self):
        batch_size = self.config.batch_gpu
        z1 = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        return z1, z2

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch

        # sample noise
        z = torch.randn(images.shape[0], self.generator.latent_dim)
        z = z.type_as(images)

        ones = torch.ones(images.shape[0], 1)
        ones.type_as(images)
        zeros = torch.zeros(images.shape[0], 1)
        zeros.type_as(images)

        # generator
        if optimizer_idx == 0:
            w = self.generator.mapping_network(z)
            # TODO style mixing
            generated_images = self.generator.synthesis_network(w)
            d_outputs = self.discriminator(generated_images)
            g_loss = self.adversarial_criterion(d_outputs, ones)

            return g_loss

        # discriminator
        if optimizer_idx == 1:
            d_real_outputs = self.discriminator(images)
            real_loss = self.adversarial_criterion(d_real_outputs, ones)

            w = self.generator.mapping_network(z)
            generated_images = self.generator.synthesis_network(w)
            d_fake_outputs = self.discriminator(generated_images)
            fake_loss = self.adversarial_criterion(d_fake_outputs, zeros)

            d_loss = real_loss + fake_loss

            return d_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        g_opt = Adam(self.generator.parameters(), lr=self.lr)
        d_opt = Adam(self.discriminator.parameters(), lr=self.lr)

        return g_opt, d_opt
