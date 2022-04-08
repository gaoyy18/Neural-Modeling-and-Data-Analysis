
import torch
import torch.nn as nn
import math
from torchsummary import summary


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=11, stride=stride,
                               padding=5, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(.2)
        # SE
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_down = nn.Conv1d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv1d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)


        if self.downsample is not None:
            residual = self.downsample(x)


        res = out1 * out + residual
        res = self.relu(res)

        return res


class SEResNet(nn.Module):

    def __init__(self, block, layers, num_classes=34):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion + 2, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data, age, gender):
        x = self.conv1(data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        age = torch.unsqueeze(age, 1)
        gender = torch.unsqueeze(gender, 1)
        inp = torch.cat((x, age, gender), 1)

        x = self.fc(inp)

        return x


def SEResNet50(**kwargs):
    model = SEResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def SEResNet101(**kwargs):
    model = SEResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def SEResNet152(**kwargs):
    model = SEResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def test_SEResNet():
    net = SEResNet50().cuda()
    print( summary(net, input_size=[(5, 8000), (1), (1)], batch_size=2) )


if __name__ == '__main__':
    pass
    input = torch.randn(2, 8, 2048)
    age = torch.randn(2, 1)
    gender = torch.randn(2, 1)
    net = SEResNet50()
    print(net(input, age, gender).shape)

