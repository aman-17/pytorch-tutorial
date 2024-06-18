import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(self, in_channels =3, num_classes = 1000):
        super().__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride =2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,144,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, out1, red3, out3, red5, out5, out1_pool):
        super().__init__()
        self.branch1 = conv_block(in_channels, out1, kernel_size = 1)
        self.branch2 = nn.Sequential(conv_block(in_channels, red3, kernel_size = 1),
                                     conv_block(red3, out3, kernel_size = 3, stride = 1, padding = 1))
        self.branch3 = nn.Sequential(conv_block(in_channels, red5, kernel_size = 1),
                                     conv_block(red5, out5, kernel_size = 5, stride = 1, padding = 2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                                     conv_block(in_channels, out1_pool, kernel_size = 1))
    
    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **args):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **args)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
    
if __name__ == '__main__':
    x =torch.randn(3,3,224,224)
    model = GoogLeNet()
    print(model(x).shape)