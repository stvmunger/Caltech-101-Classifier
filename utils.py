import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from config import BATCH_SIZE

class TxtDataset(Dataset):
    def __init__(self, list_file, transform=None):
        # list_file: 每行 "<path> <label>"
        lines = [l.strip().split() for l in open(list_file)]
        self.paths  = [p for p, _ in lines]
        self.labels = [int(lbl) for _, lbl in lines]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_dataloader(list_file, augment=False, batch_size=BATCH_SIZE):
    tf_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]
    if augment:
        tf_list.insert(0, transforms.RandomHorizontalFlip())
    transform = transforms.Compose(tf_list)

    ds = TxtDataset(list_file, transform=transform)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=augment, num_workers=4)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with __import__('torch').no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds  += preds
            all_labels += y.tolist()
    return classification_report(all_labels, all_preds, output_dict=True)
