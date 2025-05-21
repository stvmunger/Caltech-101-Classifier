import argparse
from train import train
from test import test
from config import BATCH_SIZE, LR_BACKBONE, LR_HEAD, NUM_EPOCHS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',       choices=['train','test'], required=True)
    p.add_argument('--backbone',   default='resnet18')
    p.add_argument('--pretrained', action='store_true',
                   help='是否使用 ImageNet 预训练权重')
    p.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    p.add_argument('--lr_backbone',type=float, default=LR_BACKBONE)
    p.add_argument('--lr_head',    type=float, default=LR_HEAD)
    p.add_argument('--epochs',     type=int,   default=NUM_EPOCHS)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args.backbone, args.pretrained,
              args.batch_size, args.lr_backbone,
              args.lr_head, args.epochs)
    else:
        test(args.backbone, args.pretrained, args.epochs)
