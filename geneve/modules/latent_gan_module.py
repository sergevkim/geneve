import pytorch_lightning as pl
import torch
from cleanfid import fid
from torch.nn import BCELoss

from geneve.models import Discriminator, LatentGenerator

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class LatentGANModule(pl.LightningModule):
    def __init__(self, config, latent_dim: int = 128):
        super().__init__()
        self.save_hyperparameters(config)
        self.latent_dim = latent_dim

        self.config = config
        self.G = LatentGenerator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3, synthesis_layer=config.generator)
        self.D = Discriminator(config.image_size, 3)

        self.adversarial_criterion = BCELoss()

    def latent(self):
        batch_size = self.config.batch_gpu
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch

        # sample noise
        z = torch.randn(images.shape[0], self.latent_dim)
        z = z.type_as(images)

        ones = torch.ones(images.shape[0], 1)
        ones.type_as(images)
        zeros = torch.zeros(images.shape[0], 1)
        zeros.type_as(images)

        # generator
        if optimizer_idx == 0:
            w = self.G.mapping_network(z)
            # TODO style mixing
            generated_images = self.G.synthesis_network(w)
            d_outputs = self.D(generated_images)

            g_loss = self.adversarial_criterion(d_outputs, ones)
            return g_loss

        # discriminator
        if optimizer_idx == 1:
            d_real_outputs = self.D(images)
            real_loss = self.adversarial_criterion(d_real_outputs, ones)

            w = self.mapping_network(z)
            generated_images = self.G(w)
            d_fake_outputs = self.D(generated_images)
            fake_loss = self.adversarial_criterion(d_fake_outputs, zeros)

            d_loss = real_loss + fake_loss

            return d_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.G.parameters()), lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

