from argparse import ArgumentParser

from geneve.datamodules import CIFAR10DataModule
from geneve.models import LatentGenerator, Discriminator
from geneve.modules import LatentGANModule
from pytorch_lightning import Trainer, seed_everything


def main(args):
    seed_everything(9, workers=True)

    generator = LatentGenerator(batch_size=args.batch_size)
    discriminator = Discriminator()
    module = LatentGANModule(generator=generator, discriminator=discriminator)
    datamodule = CIFAR10DataModule(batch_size=args.batch_size)
    trainer = Trainer()
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    args = parser.parse_args()
    main(args)
