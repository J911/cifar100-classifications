import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=stride, padding=0, bias=False)
    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if (self.stride != 1):
            shortcut = self.conv1x1(shortcut)
    
        x += shortcut
        x = self.relu(x)
        
        return x
        
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self.get_layers(block, 64, 64, 1, num_block[0])
        self.layer2 = self.get_layers(block, 64, 128, 2, num_block[1])
        self.layer3 = self.get_layers(block, 128, 256, 2, num_block[2])
        self.layer4 = self.get_layers(block, 256, 512, 2, num_block[3])
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def get_layers(self, block, in_channels, out_channels, stride, num_block):
        layers = []
        
        for i in range(num_block):
            if i == 0:
                layers.append(block(in_channels, out_channels, stride))
                continue
            layers.append(block(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
                          
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def resnet18(num_classes=100):
    return ResNet(Block, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=100):
    return ResNet(Block, [3, 4, 6, 3], num_classes)