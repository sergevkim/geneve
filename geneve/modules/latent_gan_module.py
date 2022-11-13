from pathlib import Path

import cv2
import einops
import pytorch_lightning as pl
import torch
import torchvision
from cleanfid import fid
from torch.nn import BCELoss
from torch.optim import Adam


class LatentGANModule(pl.LightningModule):
    def __init__(self, generator, discriminator, lr: float = 3e-4):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.adversarial_criterion = BCELoss()
        self.lr = lr
        self.generated_images_dir = Path('data/generated')
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)

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
        w = self.generator.mapping_network(z)
        generated_images = self.generator.synthesis_network(w)

        ones = torch.ones(images.shape[0], 1).to(images)
        ones.type_as(images)
        zeros = torch.zeros(images.shape[0], 1).to(images)
        zeros.type_as(images)

        # generator
        if optimizer_idx == 0:
            # TODO style mixing
            d_outputs = self.discriminator(generated_images)
            g_loss = self.adversarial_criterion(d_outputs, ones)

            return g_loss

        # discriminator
        if optimizer_idx == 1:
            d_real_outputs = self.discriminator(images)
            real_loss = self.adversarial_criterion(d_real_outputs, ones)
            d_fake_outputs = self.discriminator(generated_images)
            fake_loss = self.adversarial_criterion(d_fake_outputs, zeros)
            d_loss = real_loss + fake_loss

            return d_loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        images, _ = batch
        if batch_idx == 0:
            z = torch.randn(images.shape[0], self.generator.latent_dim)
            z = z.type_as(images)
            w = self.generator.mapping_network(z)
            generated_images = self.generator.synthesis_network(w)
            images_grid = \
                torchvision.utils.make_grid(generated_images).cpu().numpy()
            images_grid = einops.rearrange(images_grid, 'c h w -> h w c')
            #self.log() TODO log images

        for image_idx, image in enumerate(images):
            cv2.imwrite(
                f'{self.generated_images_dir}/{batch_idx}_{image_idx}',
                image,
            )


    def validation_epoch_end(self):
        fid_score = fid.compute_fid(
            self.generated_images_dir,
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="test",
        )
        self.log(fid_score)


    def configure_optimizers(self):
        g_opt = Adam(self.generator.parameters(), lr=self.lr)
        d_opt = Adam(self.discriminator.parameters(), lr=self.lr)

        return g_opt, d_opt
