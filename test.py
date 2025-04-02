import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import create_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def test_model(model_path, test_dir, num_classes=75, device='cuda'):
    """
    Test the trained model and generate performance metrics
    
    Args:
        model_path (str): Path to saved model weights
        test_dir (str): Directory containing test images
        num_classes (int): Number of output classes
        device (str): Device to run testing on
    
    Returns:
        dict: Performance metrics
    """
    # Test data transformation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and load model
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Prediction and ground truth storage
    all_preds = []
    all_labels = []
    
    # Test phase
    with torch.no_grad():
        correct = 0
        total = 0
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Generate classification report
    class_names = test_dataset.classes
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def main():
    # Configuration
    model_path = 'best_model.pth'
    test_dir = r"C:\Python\AI_APP\dataset\Test"  # Directly point to Test folder
    num_classes = 75
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run test
    results = test_model(model_path, test_dir, num_classes, device)
    
    # Print results
    print(f"Test Accuracy: {results['accuracy']}%")
    print("\nClassification Report:")
    print(results['classification_report'])

if __name__ == '__main__':
    main()