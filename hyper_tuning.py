import itertools
from train import train
from config import BATCH_SIZE, LR_BACKBONE, LR_HEAD, NUM_EPOCHS

grid = {
    'lr_head':     [1e-2, 1e-3],
    'num_epochs':  [20, 30]
}

for lr, epochs in itertools.product(grid['lr_head'], grid['num_epochs']):
    print(f"--- LR_HEAD={lr}, EPOCHS={epochs} ---")
    train('resnet18', True,
          batch_size=BATCH_SIZE,
          lr_backbone=LR_BACKBONE,
          lr_head=lr,
          num_epochs=epochs)
