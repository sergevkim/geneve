from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from geneve.datamodules import CIFAR10DataModule
from geneve.models import Discriminator, LatentGenerator
from geneve.modules import LatentGANModule


def main(args):
    seed_everything(9, workers=True)

    generator = LatentGenerator(height=args.height, width=args.width)
    discriminator = Discriminator()
    module = LatentGANModule(generator=generator, discriminator=discriminator)
    datamodule = CIFAR10DataModule(batch_size=args.batch_size)

    logger = WandbLogger(project='geneve')
    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=2,
    #     monitor='fid',
    #     mode='max',
    #     dirpath='checkpoints',
    #     filename='medium-latent-{epoch:02d}-{fid:.2f}',
    # )
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=40,
        accelerator='gpu',
        logger=logger,
        #callbacks=[checkpoint_callback],
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    args = parser.parse_args()
    main(args)
