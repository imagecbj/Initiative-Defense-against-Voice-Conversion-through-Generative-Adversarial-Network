from hyperparameter import hp
from .dis import Discriminator
from .gen import Generator
from .swcsm import SWCSM


def load_gen_and_dis():
    G = Generator().to(hp.device)
    D = Discriminator().to(hp.device)
    return G, D
