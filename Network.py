import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0, groups=1, dilation=1, bias=bias)

    def forward(self,img):
        x = self.depthwise(img)
        x = self.pointwise(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, first_relu=True):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.layers = [
            self.relu,
            SeparableConv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            self.relu,
            SeparableConv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]

        if first_relu == False:
            self.layers = self.layers[1:]

        self.mainstream = nn.Sequential(*self.layers)

    def forward(self, img):
        x = self.mainstream(img)
        x += self.shortcut(img)

        return x


class MiddleFlow(nn.Module):
    def __init__(self, num_channels, stride=3, repeat=8):
        super(MiddleFlow, self).__init__()

        self.repeat = repeat
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        for i in range(3):
            self.layers += [
                self.relu,
                SeparableConv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(num_channels),
            ]
        self.mainstream = nn.Sequential(self.layers)

    def forward(self, img):
        x = img
        for i in range(self.repeat):
            x += self.mainstream(x)

        return x


class XceptionNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(XceptionNet, self).__init__()

        self.num_classes = num_classes

        self.entry_layers = [
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        self.entryBlock = nn.Sequential(*self.entry_layers)

        self.resBlock1 = ResBlock(64, 128, 128, first_relu=False)
        self.resBlock2 = ResBlock(128, 256, 256)
        self.resBlock3 = ResBlock(256, 728, 728)

        self.middleFlow = MiddleFlow(num_channels=728, stride=3, repeat=8)

        self.resBlock4 = ResBlock(728, 728, 1024)

        self.exit_layers = [
            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        ]

        self.exitBlock = nn.Sequential(*self.exit_layers)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img):
        x = self.entryBlock(img)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.middleFlow(x)
        x = self.resBlock4(x)
        x = self.exitBlock(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = self.fc(x)

        return x
















