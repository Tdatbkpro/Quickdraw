import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomAffine
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from src.dataset import QuickDrawDataset
from src.model import QuickDraw
from src.config import CLASSES

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--optimizer", "-op", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--total_images-per-class", "-ipc", type=int, default=24000)
    parser.add_argument("--ratio", "-r", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-workers", "-nw", type=int, default=4)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--img-size", "-is", type=int, default=28)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--checkpoint", "-cp", type=str, default="trained_models/last.pt")
    parser.add_argument("--data_path", type=str, default="dataset_quickdraw")
    return parser.parse_args()

def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = Compose([
        ToTensor(),
        RandomAffine(degrees=(-10, 10), translate=(0.08, 0.08), scale=(0.95, 1.05), shear=5),
    ])
    transform_test = ToTensor()

    train_dataset = QuickDrawDataset(
        root_path=args.data_path,
        total_images_per_class=args.total_images_per_class,
        ratio=args.ratio,
        mode="train",
        transform=transform_train
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)

    test_dataset = QuickDrawDataset(
        root_path=args.data_path,
        total_images_per_class=args.total_images_per_class,
        ratio=args.ratio,
        mode="test",
        transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    model = QuickDraw(input_size=args.img_size, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" \
        else SGD(model.parameters(), lr=args.lr, momentum=0.9)

    os.makedirs(args.saved_path, exist_ok=True)
    if os.path.exists(args.log_path): shutil.rmtree(args.log_path)
    writer = SummaryWriter(args.log_path)

    start_epoch, best_acc = 0, 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", colour="green")
        for i, (images, labels) in enumerate(progress):
            images, labels = images.to(device), labels.to(device)
            # print(images[1])
            # image_tensor = images[1].cpu()  # chuyển về CPU nếu đang ở GPU
            # label = labels[1].item()  
            # plt.imshow(image_tensor.squeeze(0), cmap="gray")
            # plt.title(f"Label: {label}")
            # plt.axis("off")
            # plt.show()
            # print(labels[1])
            # exit(0)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)
            progress.set_postfix(loss=loss.item())

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")
        writer.add_scalar("Test/Accuracy", acc, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }
        torch.save(checkpoint, os.path.join(args.saved_path, "last.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.saved_path, "best.pt"))
            best_acc = acc

if __name__ == "__main__":
    train()
