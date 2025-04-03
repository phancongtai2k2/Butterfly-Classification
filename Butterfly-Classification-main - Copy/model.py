import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=75):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of the feature maps after convolutions
        def _get_conv_output(input_size):
            return torch.zeros(1, 3, input_size, input_size)
        
        with torch.no_grad():
            x = self._get_conv_output(224)
            conv_out = self.conv_layers(x)
            flattened_size = conv_out.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_conv_output(self, input_size):
        return torch.zeros(1, 3, input_size, input_size)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def create_model(num_classes=75, pretrained=False):
    """
    Create the CNN model with optional pretraining
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Configured CNN model
    """
    model = CNNModel(num_classes)
    
    # Optional: Add pretrained weights loading logic here
    if pretrained:
        # Load pretrained weights if available
        pass
    
    return model