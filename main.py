import warnings
from trainer import Trainer

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    t = Trainer()
    t.train()
