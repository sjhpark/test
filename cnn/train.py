import numpy as np
import torch
import torchvision
from tqdm import tqdm
from model import CNN
from torch.utils.data import DataLoader
from utils import loaddata
import argparse

def train(epochs, batch_size, lr):
    # dataset
    train_dataset, val_dataset = loaddata()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # sample image
    sample, _ = next(iter(train_loader))
    input_channels = sample.shape[1]
    num_classes = len(train_dataset.classes)
    print(f"Input channels: {input_channels}, Num classes: {num_classes}")

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(input_channels, num_classes).to(device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TRAINING
    for epoch in tqdm(range(epochs), desc='Training'):
        model.train()
        tot_loss = 0

        # ITERATION OVER BATCHES
        for i, data in tqdm(enumerate(train_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)

            # ZERO OUT GRADIENTS
            optimizer.zero_grad()

            # FORWARD PASS
            pred = model(x)
            
            # COMPUTE LOSS (ERROR)
            loss = criterion(pred, y)

            # BACKPROP
            loss.backward()

            # UPDATE WEIGHTS
            optimizer.step()

            tot_loss += loss.item()

        # print avg loss per epoch
        avg_loss = tot_loss / len(train_loader)
        print(f'Epoch: {epoch}, Train Loss: {avg_loss}')
            

        # VALIDATION
        model.eval()
        with torch.no_grad(): # no need to update weights (parameters)
            correct = 0
            tot_loss = 0
            total = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                tot_loss += loss.item()
                _, pred_indices = torch.max(pred, 1)
                total += y.size(0)
                correct += (pred_indices == y).sum().item()
            print(f'Epoch: {epoch}, Val Loss: {tot_loss/len(val_loader)}')
            print(f'Epoch: {epoch}, Val Accuracy: {100*correct/total}%')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, required=True)
    argparser.add_argument('--batch', type=int, required=True)
    argparser.add_argument('--lr', type=float, required=True)
    args = argparser.parse_args()

    train(args.epochs, args.batch, args.lr)


