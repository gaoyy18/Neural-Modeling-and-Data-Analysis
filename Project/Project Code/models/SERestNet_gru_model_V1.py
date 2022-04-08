import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['SERestNet_gru_model']


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
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, data):
        x = self.conv1(data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print('before avgpool:', x.shape)
        x = self.avgpool(x)
        # print('after avgpool:', x.shape)
        x = x.view(x.size(0), -1)

        # age = torch.unsqueeze(age, 1)
        # gender = torch.unsqueeze(gender, 1)
        # inp = torch.cat((x, age, gender), 1)

        # x = self.fc(x)

        return x


def SEResNet50_Basic(**kwargs):
    model = SEResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class GRU_Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, n_layers, dropout=0.3, bidirectional=True):
        super(GRU_Layer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1, n_layers, dropout=dropout, bidirectional=bidirectional)

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        self.gru.flatten_parameters()
        return self.gru(x)


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        x = x.transpose(0, 1).clone()

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = torch.bmm(x.transpose(1, 2), torch.unsqueeze(a, -1))

        return torch.sum(weighted_input, 1)


class GRU0(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):  # , output_size
        super(GRU0, self).__init__()

        self.seq_length = int(2048 / input_size)
        self.n_layers = 2

        self.rnn = GRU_Layer(input_size, hidden_size1, self.n_layers, dropout=0.1, bidirectional=True)
        self.rnn.init_weights()
        self.attention = Attention(hidden_size1 * 2, self.seq_length)
        self.r2o = nn.Linear(hidden_size1 * 2 * 3 + 1, hidden_size2)
        self.bn = nn.BatchNorm1d(hidden_size2, momentum=0.9)
        self.T = int(2048 / input_size)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        weight = self.attention(hidden)
        # global average pooling
        avg_pool = torch.mean(hidden, 0)
        # global max pooling
        max_pool, _ = torch.max(hidden, 0)
        output = torch.cat((hidden[self.T - 1], weight, avg_pool, max_pool), 1)
        output = F.relu(self.r2o(output))
        output = self.bn(output)

        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = F.relu(self.i2h(input))
        output = self.h2o(hidden)

        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes=34):
        super(RNN, self).__init__()

        self.resnet = SEResNet50_Basic()
        self.rnn = GRU0(input_size, hidden_size1, hidden_size2)

        self.batch = int(256 / 4)
        self.total_len = 5000
        self.input_size = input_size
        self.seq_len = int(self.total_len / input_size)

        self.mlp = MLP(hidden_size2 + 2, hidden_size3, num_classes)

    def forward(self, x, age, gender):

        x = self.resnet(x)

        self.total_len = x.size()[1]
        self.seq_len = int(self.total_len / self.input_size)
        out = self.rnn(x.view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        rnn_out = torch.cat((out, age.view(self.batch, 1), gender.view(self.batch, 1)), 1)

        output = self.mlp(rnn_out)

        return output


def SERestNet_gru_model(pretrained=False, **kwargs):
    """Constructs a rnn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    feature_dim = 64
    hidden_size1 = feature_dim
    hidden_size2 = 250
    hidden_size3 = 100
    # feature_dim ——> hidden_size1=feature_dim*2
    # feature_dim*6+1 ——> hidden_size2 ——> hidden_size2 * 8(concat)
    # hidden_size2 * 8 ——> hidden_size3 ——> num_classes=34
    model = RNN(feature_dim, hidden_size1, hidden_size2, hidden_size3, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model