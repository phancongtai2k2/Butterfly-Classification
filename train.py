import os
import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast

from model import create_model

import os
import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast

from model import create_model

def create_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for training, validation, and testing
    """
    # Scan for actual folder names (case-insensitive)
    folders = {
        folder.lower(): folder 
        for folder in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, folder))
    }
    
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Map possible variations of folder names
    name_mappings = {
        'train': ['train', 'trains', 'training'],
        'val': ['val', 'validation', 'valid'],
        'test': ['test', 'tests', 'testing']
    }
    
    # Find matching folders
    image_datasets = {}
    for key, variations in name_mappings.items():
        for variation in variations:
            if variation in folders:
                image_datasets[key] = datasets.ImageFolder(
                    os.path.join(data_dir, folders[variation]), 
                    data_transforms[key]
                )
                break
        
        if key not in image_datasets:
            raise ValueError(f"Could not find {key} folder. Available folders: {list(folders.keys())}")
    
    # Create data loaders
    data_loaders = {
        key: DataLoader(
            image_datasets[key], 
            batch_size=batch_size, 
            shuffle=key == 'train',
            num_workers=4
        )
        for key in image_datasets
    }
    
    return data_loaders, image_datasets

# Rest of the code remains the same as in the previous version

def train_model(model, dataloaders, datasets, criterion, optimizer, num_epochs=40, device='cuda'):
    """
    train the CNN model
    
    Args:
        model (torch.nn.Module): Model to train
        dataloaders (dict): Dictionary of data loaders
        datasets (dict): Dictionary of datasets
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        device (str): Device to train on (cuda/cpu)
    
    Returns:
        torch.nn.Module: trained model
    """
    model.to(device)
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with autocasting
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        # validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'training Loss: {running_loss/len(dataloaders["train"]):.4f}')
        print(f'validation Loss: {val_loss/len(dataloaders["val"]):.4f}')
        print(f'validation Accuracy: {100 * correct / total:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 40
    num_classes = 75
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(num_classes)
    
    # Data loaders
    data_loaders, datasets = create_data_loaders("C:/Python/AI_APP/dataset", batch_size)
    
    # Loss and optimizer
    criterion = torch_nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # train the model
    trained_model = train_model(
        model, 
        data_loaders, 
        datasets,
        criterion, 
        optimizer, 
        num_epochs, 
        device
    )

if __name__ == '__main__':
    main()