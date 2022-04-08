import torch.nn as nn
import torch

def conv_2d(in_planes, out_planes, stride=(1,1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,size), stride=stride,
                     padding=(0,(size-1)//2), bias=False)

def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size-1)//2, bias=False)

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

        # SE
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_down = nn.Conv1d(
            planes, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv1d(
            planes // 4, planes, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

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

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            res = out1 * out + residual
            res = self.relu(res)
        
        return res

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

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

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            res = out1 * out + residual
            res = self.relu(res)
        
        return res



  
class ECGNet_V2(nn.Module):
    def __init__(self, input_channel=1, num_classes=34):
        # 卷积核大小
        sizes = [3, 5, 7]
        self.sizes = sizes
        # 叠加block个数, conv2d在前，conv1d在后
        layers = [6, 8]
           

        super(ECGNet_V2, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1,50), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.inplanes = 32 
            self.inplanes = 32 
            self.layers1 = self._make_layer2d(BasicBlock2d, 32, layers[0], stride=(1,1), size=size)

            self.inplanes *= 8
            self.layers2 = self._make_layer1d(BasicBlock1d, 256, layers[1], stride=2, size=size)
            
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256*len(sizes)+2, num_classes)
        
    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)
    
    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True):
        downsample = None
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)
    

    def forward(self, x0, age, gender):
        x0 = x0.unsqueeze(1)

        x0 = self.conv1(x0)     
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x0 = self.maxpool(x0)

        x0 = self.conv2(x0)
        x0 = self.bn2(x0)
        x0 = self.relu(x0)

        x0 = self.maxpool(x0)
        #x0 = self.dropout(x0)
        

        xs = []
        for i in range(len(self.sizes)):
            # print(self.layers1_list[i])
            # print("------")
            # print(self.layers2_list[i])
            # print("------")
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1,end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)

        age = torch.unsqueeze(age, 1)
        gender = torch.unsqueeze(gender, 1)
        inp = torch.cat((out, age, gender), 1)
        out = self.fc(inp)

        return out

def ECGNet_v2():
    return ECGNet_V2()

if __name__ == '__main__':
    pass
    input = torch.randn(2, 8, 5000)
    age = torch.randn(2, 1)
    gender = torch.randn(2, 1)
    net = ECGNet_V2()
    print(net(input, age, gender).shape)