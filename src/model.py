"""
src/model.py - 7-Class Neural Network Architecture for Rice Field State Classification
Outputs: Dry, Water, Wet, Green rice, Green weed, Straw, Others.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 7 target categories in Dataset/
CLASSES = [
    "Dry",
    "Water",
    "Wet",
    "Green rice",
    "Green weed",
    "Straw",
    "Others"
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

CLASS_COLORS = {
    "Dry": (218, 195, 60),        # Golden/Yellow for Dry
    "Water": (0, 119, 182),       # Deep water blue
    "Wet": (70, 130, 180),        # Muddy/slate blue for wet soil
    "Green rice": (46, 204, 113), # Vivid lime/emerald green for rice crops
    "Green weed": (34, 139, 34),  # Dark forest green for weeds
    "Straw": (225, 190, 100),     # Golden straw / dry mulch
    "Others": (120, 80, 70),      # Dark neutral/brown for Others
}


class ResidualBlock(nn.Module):
    """Residual convolutional block with skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class RiceFieldClassifier(nn.Module):
    """
    Deep Residual CNN for classifying rice fields into 7 states:
    [Dry, Water, Wet, Green rice, Green weed, Straw, Others]
    """
    def __init__(self, num_classes=len(CLASSES), in_channels=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        dropped = self.dropout(features)
        logits = self.fc(dropped)
        
        if return_features:
            return logits, features
        return logits


def build_classifier(num_classes=len(CLASSES)):
    """Helper to instantiate the rice field classifier."""
    return RiceFieldClassifier(num_classes=num_classes)


if __name__ == "__main__":
    model = build_classifier()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Model test successful! Output shape:", out.shape)
