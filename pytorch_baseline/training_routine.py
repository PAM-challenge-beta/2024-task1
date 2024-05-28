import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pytorch_baseline.model import CNNModel
from pytorch_baseline.dataloader import HDF5Dataset

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(hdf5_db, input_shape=(128,128,1), num_classes=2, train_table="/train", val_table=None, epochs=10, batch_size=32, seed=None, device=default_device):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    train_dataset = HDF5Dataset(hdf5_db, train_table, num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_table is not None:
        val_dataset = HDF5Dataset(hdf5_db, val_table, num_classes)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    model = CNNModel(input_shape=input_shape, num_classes=num_classes, device = device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.to(device)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = 100 * correct / total
        losses.append(train_loss)

        if val_table is not None:
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    
    return model, losses


def validate_model(model, val_loader, criterion, device=default_device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()