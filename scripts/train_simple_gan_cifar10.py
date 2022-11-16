from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from geneve.datamodules import CIFAR10DataModule
from geneve.models import Discriminator, SimpleGenerator
from geneve.modules import SimpleGANModule


def main(args):
    seed_everything(9, workers=True)

    generator = SimpleGenerator()
    discriminator = Discriminator()
    module = SimpleGANModule(generator=generator, discriminator=discriminator)
    datamodule = CIFAR10DataModule(batch_size=args.batch_size)

    logger = WandbLogger(project='geneve')
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=40,
        accelerator='gpu',
        logger=logger,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    main(args)
