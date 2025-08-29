"""
Minimal fine-tuning stub.

Expected dataset structure:
data/
  train/
    normal/
    pneumonia/
    covid19/
    pulmonary_embolism/
  val/
    normal/
    pneumonia/
    covid19/
    pulmonary_embolism/

Run:
  python -m app.train_stub --data ./data --epochs 2 --out models/model.pt
"""
import argparse, os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

CLASSES = ["normal", "pneumonia", "covid19", "pulmonary_embolism"]

def get_loaders(data_root, batch_size=16):
    tfm_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root,"train"), transform=tfm_train)
    val_ds = datasets.ImageFolder(os.path.join(data_root,"val"), transform=tfm_val)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dl, val_dl

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dl, val_dl = get_loaders(args.data, batch_size=args.batch_size)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
        train_acc = correct/total
        train_loss = loss_sum/total

        # val
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total += x.size(0)
        val_acc = correct/total if total>0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print("Saved", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", type=str, default="models/model.pt")
    main(p.parse_args())
