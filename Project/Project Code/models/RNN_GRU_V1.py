import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['gru_channel8_model']


class GRU0(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2):
        super(GRU0, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_szie1, 2, dropout=0.3, bidirectional=True)
        self.r2o = nn.Linear(hidden_szie1 * 2, hidden_szie2)
        self.T = 10

    def forward(self, input):
        self.rnn.flatten_parameters()
        hidden, _ = self.rnn(input)
        output = F.relu(self.r2o(hidden[self.T - 1]))

        return output

class MLP(nn.Module):
    def __init__(self, input_size, hidden_szie, output_size):
        super(MLP, self).__init__()

        self.i2h = nn.Linear(input_size, hidden_szie)
        self.h2o = nn.Linear(hidden_szie, output_size)

    def forward(self, input):
        hidden = F.relu(self.i2h(input))
        output = self.h2o(hidden)

        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes=34):
        super(RNN, self).__init__()

        self.rnn1=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn2=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn3=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn4=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn5=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn6=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn7=GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn8=GRU0(input_size, hidden_size1, hidden_size2)

        self.batch = 128

        self.mlp = MLP(hidden_size2 * 8, hidden_size3, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1).clone()

        out1 = self.rnn1(x[0].view(self.batch,10,500).transpose(0,1))
        out2 = self.rnn2(x[1].view(self.batch,10,500).transpose(0,1))
        out3 = self.rnn3(x[2].view(self.batch,10,500).transpose(0,1))
        out4 = self.rnn4(x[3].view(self.batch,10,500).transpose(0,1))
        out5 = self.rnn5(x[4].view(self.batch,10,500).transpose(0,1))
        out6 = self.rnn6(x[5].view(self.batch,10,500).transpose(0,1))
        out7 = self.rnn7(x[6].view(self.batch,10,500).transpose(0,1))
        out8 = self.rnn8(x[7].view(self.batch,10,500).transpose(0,1))
        rnn_out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), 1)
        output = self.mlp(rnn_out)
        return output


def gru_channel8_model(pretrained=False, **kwargs):
    """Constructs a rnn_gru model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    feature_dim = 500
    hidden_size1 = feature_dim
    hidden_size2 = 250  # 200#250
    hidden_size3 = 100
    # feature_dim ——> hidden_size1=feature_dim*2
    # feature_dim*6+1 ——> hidden_size2
    # hidden_size2 ——> hidden_size3 ——> num_classes=34
    model = RNN(feature_dim, hidden_size1, hidden_size2, hidden_size3, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model