import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from config import *
from utils import get_dataloader, evaluate
from model import get_model
from scripts.split_dataset import split_dataset

def train(backbone, pretrained, batch_size, lr_backbone, lr_head, num_epochs):
    # 1. 划分数据集
    if not os.path.exists(TRAIN_LIST):
        split_dataset(DATA_DIR, TRAIN_LIST, TEST_LIST, TRAIN_SAMPLES)

    # 2. DataLoader
    train_loader = get_dataloader(TRAIN_LIST, augment=True,  batch_size=batch_size)
    val_loader   = get_dataloader(TEST_LIST,  augment=False, batch_size=batch_size)

    # 3. 模型 & 优化器
    model = get_model(backbone, pretrained)
    backbone_params = [p for n,p in model.named_parameters() if 'fc' not in n]
    head_params     = [p for n,p in model.named_parameters() if 'fc' in n]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head}
    ], weight_decay=WEIGHT_DECAY)

    # 4. TensorBoard 写入器
    tag = f"{backbone}_{'pre' if pretrained else 'scratch'}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, tag))

    # 5. 训练循环
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar('train/loss', loss.item(), epoch)

        report = evaluate(model, val_loader, DEVICE)
        val_acc = report['accuracy']
        writer.add_scalar('val/accuracy', val_acc, epoch)

        print(f"[{tag}] Epoch {epoch}/{num_epochs}  "
              f"Loss: {total_loss/len(train_loader):.4f}  Acc: {val_acc:.4f}")

        if epoch % SAVE_FREQ == 0:
            fn = f"{tag}_e{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, fn))

    writer.close()


if __name__ == '__main__':
    # 方便调试
    train('resnet18', True, BATCH_SIZE, LR_BACKBONE, LR_HEAD, NUM_EPOCHS)
