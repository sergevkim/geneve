from pathlib import Path

import cv2
import einops
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from cleanfid import fid
from torch.nn import BCELoss, Module
from torch.optim import Adam


class SimpleGANModule(pl.LightningModule):
    def __init__(
        self,
        generator: Module,
        discriminator: Module,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.adversarial_criterion = BCELoss()
        self.lr = lr
        self.generated_images_dir = Path('data/generated')
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch

        # sample noise
        bs, channels, height, width = images.shape
        w = torch.randn((bs, 128, 1, 1))
        generated_images = self.generator(w)

        ones = torch.ones(images.shape[0], 1).to(self.device)
        zeros = torch.zeros(images.shape[0], 1).to(self.device)

        # generator
        if optimizer_idx == 0:
            d_outputs = self.discriminator(generated_images)
            g_loss = self.adversarial_criterion(d_outputs, ones)
            self.log('g_loss', g_loss)

            return g_loss

        # discriminator
        if optimizer_idx == 1:
            if batch_idx % 5 == 0:
                d_real_outputs = self.discriminator(images)
                real_loss = self.adversarial_criterion(d_real_outputs, ones)
                d_fake_outputs = self.discriminator(generated_images)
                fake_loss = self.adversarial_criterion(d_fake_outputs, zeros)
                d_loss = real_loss + fake_loss
                self.log('d_loss', d_loss)

                return d_loss

    def validation_step(self, batch, batch_idx):
        pass

    @torch.inference_mode()
    def validation_epoch_end(self, val_step_outputs):
        w = torch.randn(16, 128, 1, 1).to(self.device)
        generated_images = self.generator(w)
        images_grid = \
            torchvision.utils.make_grid(generated_images).cpu().numpy()
        images_grid = einops.rearrange(images_grid, 'c h w -> h w c')
        self.logger.experiment.log({"images": [wandb.Image(images_grid)]})

        for batch_idx in range(40):
            w = torch.randn(250, 128, 1, 1).to(self.device)
            generated_images = self.generator(w).cpu().numpy()
            generated_images = \
                einops.rearrange(generated_images, 'bs c h w -> bs h w c')

            for image_idx, image in enumerate(generated_images):
                cv2.imwrite(
                    f'{self.generated_images_dir}/{batch_idx}_{image_idx}.png',
                    image,
                )

        fid_score = fid.compute_fid(
            str(self.generated_images_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="test",
        )
        self.log('fid_score', fid_score)

    def configure_optimizers(self):
        g_opt = Adam(self.generator.parameters(), lr=self.lr)
        d_opt = Adam(self.discriminator.parameters(), lr=self.lr)

        return g_opt, d_opt
