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
        lr: float = 0.0002,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.automatic_optimization = False
        self.adversarial_criterion = BCELoss()
        self.lr = lr

        self.generated_images_dir = Path('data/generated')
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        images, _ = batch

        # sample noise
        bs = images.shape[0]
        w = torch.randn((bs, 128, 1, 1)).to(self.device)
        generated_images = self.generator(w)

        ones = torch.ones(images.shape[0], 1).to(self.device)
        zeros = torch.zeros(images.shape[0], 1).to(self.device)

        # discriminator
        d_opt.zero_grad()
        d_real_outputs = self.discriminator(images)
        real_loss = self.adversarial_criterion(d_real_outputs, ones)
        self.manual_backward(real_loss)
        d_fake_outputs = self.discriminator(generated_images.detach())
        fake_loss = self.adversarial_criterion(d_fake_outputs, zeros)
        self.manual_backward(fake_loss)
        d_loss = real_loss + fake_loss
        self.log('d_loss', d_loss)
        self.log('d_real_acc', d_real_outputs.argmax().sum())
        self.log('d_fake_acc', d_fake_outputs.argmax().sum())
        d_opt.step()

        # generator
        g_opt.zero_grad()
        d_outputs = self.discriminator(generated_images)
        g_loss = self.adversarial_criterion(d_outputs, ones)
        self.log('g_loss', g_loss)
        self.manual_backward(g_loss)
        g_opt.step()

    def validation_step(self, batch, batch_idx):
        pass

    @torch.inference_mode()
    def validation_epoch_end(self, val_step_outputs):
        w = torch.randn(16, 128, 1, 1).to(self.device)
        generated_images = self.generator(w)
        images_grid = \
            torchvision.utils.make_grid(generated_images).cpu().numpy()
        images_grid = einops.rearrange(images_grid, 'c h w -> h w c') * 256
        self.logger.experiment.log({"images": [wandb.Image(images_grid)]})

        # for batch_idx in range(40):
        #     w = torch.randn(250, 128, 1, 1).to(self.device)
        #     generated_images = self.generator(w).cpu().numpy()
        #     generated_images = \
        #         einops.rearrange(generated_images, 'bs c h w -> bs h w c')

        #     for image_idx, image in enumerate(generated_images):
        #         cv2.imwrite(
        #             f'{self.generated_images_dir}/{batch_idx}_{image_idx}.png',
        #             image,
        #         )

        # fid_score = fid.compute_fid(
        #     str(self.generated_images_dir),
        #     dataset_name="cifar10",
        #     dataset_res=32,
        #     dataset_split="test",
        # )
        # self.log('fid_score', fid_score)

    def configure_optimizers(self):
        g_opt = Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        return g_opt, d_opt
