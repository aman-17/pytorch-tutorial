import torch
import torch.nn as nn

VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv(VGG16)
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,num_classes))
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),
                                   stride=(1,1), padding=(1,1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers+=[nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]        
        return nn.Sequential(*layers)
    
model = VGG(in_channels=3, num_classes=1000)
x = torch.randn(1,3,224,224)
print(model(x).shape)


class VGG2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG2, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1a = nn.BatchNorm2d(num_features=64)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1b = nn.BatchNorm2d(num_features=64)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2a = nn.BatchNorm2d(num_features=128)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2b = nn.BatchNorm2d(num_features=128)
        
        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3a = nn.BatchNorm2d(num_features=256)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3b = nn.BatchNorm2d(num_features=256)
        
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4a = nn.BatchNorm2d(num_features=512)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4b = nn.BatchNorm2d(num_features=512)
        
        self.fc1 = nn.Linear(512 * 14 * 14, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = VGG2()
x1 = torch.randn(1, 3, 224, 224)
print(model(x1).shape) 





