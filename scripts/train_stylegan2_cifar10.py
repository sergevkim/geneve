from argparse import ArgumentParser

from geneve.datamodules import CIFAR10DataModule
from geneve.models import MLPModel
from geneve.modules import StyleGAN2Module
from pytorch_lightning import Trainer, seed_everything


def main(args):
    seed_everything(9, workers=True)

    mapping_network = MLPModel()
    synthesis_network =
    discriminator = None
    module = StyleGAN2Module()
    datamodule = CIFAR10DataModule()
    trainer = Trainer()
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument()
    args = parser.parse_args()
    main(args)
