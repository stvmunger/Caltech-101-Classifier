import os
import torch
from config import *
from utils import get_dataloader, evaluate
from model import get_model

def test(backbone, pretrained, epoch):
    tag = f"{backbone}_{'pre' if pretrained else 'scratch'}"
    fn  = f"{tag}_e{epoch}.pth"
    path = os.path.join(MODEL_DIR, fn)
    assert os.path.exists(path), f"Checkpoint not found: {path}"

    model = get_model(backbone, pretrained)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)

    report = evaluate(model, get_dataloader(TEST_LIST, batch_size=BATCH_SIZE), DEVICE)
    from pprint import pprint
    print(f"== Evaluation: {tag}, epoch={epoch} ==")
    pprint(report)


if __name__ == '__main__':
    # 示例：分别评测微调和scratch训练
    test('resnet18', True,  NUM_EPOCHS)
    test('resnet18', False, NUM_EPOCHS)
