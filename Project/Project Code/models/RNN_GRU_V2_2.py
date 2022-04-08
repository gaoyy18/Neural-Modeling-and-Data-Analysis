import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['rnn_attention_gru_model']


# input_size = 500
# hidden_size =
# num_layers = 2
# dropout = 0.1
# bidirectional = True

class GRU_Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, n_layers, dropout=0.3, bidirectional=True):
        super(GRU_Layer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1, n_layers, dropout=0.3, bidirectional=True)

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

        self.input_size = input_size
        self.seq_length = int(5000 / input_size)
        self.n_layers = 2
        self.batch = 128

        self.attention = Attention(input_size, self.seq_length)
        self.i2h = nn.Linear(5000 + 1, 5000)
        self.rnn = GRU_Layer(input_size, hidden_size1, self.n_layers, dropout=0.3, bidirectional=True)
        self.rnn.init_weights()
        self.r2o = nn.Linear(hidden_size1 * 2 * 3, hidden_size2)
        self.bn = nn.BatchNorm1d(hidden_size2, momentum=0.9)

    def forward(self, input):
        weight = self.attention(input)
        out = torch.cat((input.transpose(0, 1).view(self.batch, 5000), weight), 1)
        out = self.i2h(out)
        out = out.view(self.batch, self.seq_length, self.input_size).transpose(0, 1)
        hidden, _ = self.rnn(out)
        # global average pooling
        avg_pool = torch.mean(hidden, 0)
        # global max pooling
        max_pool, _ = torch.max(hidden, 0)
        # exit(0)
        output = torch.cat((hidden[self.seq_length - 1], avg_pool, max_pool), 1)
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

        self.rnn1 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn2 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn3 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn4 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn5 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn6 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn7 = GRU0(input_size, hidden_size1, hidden_size2)
        self.rnn8 = GRU0(input_size, hidden_size1, hidden_size2)

        self.batch = 128
        self.total_len = 5000
        self.input_size = input_size
        self.seq_len = int(self.total_len / input_size)
        self.mlp = MLP(hidden_size2 * 8, hidden_size3, num_classes)

    def forward(self, x):

        x = x.transpose(0, 1).clone()

        out1 = self.rnn1(x[0].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out2 = self.rnn2(x[1].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out3 = self.rnn3(x[2].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out4 = self.rnn4(x[3].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out5 = self.rnn5(x[4].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out6 = self.rnn6(x[5].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out7 = self.rnn7(x[6].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))
        out8 = self.rnn8(x[7].view(self.batch, self.seq_len, self.input_size).transpose(0, 1))

        rnn_out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), 1)
        output = self.mlp(rnn_out)

        return output


def rnn_attention_gru_model(pretrained=False, **kwargs):
    """Constructs a rnn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    feature_dim = 500
    time_len = int(5000 / feature_dim)  # 10s
    hidden_size1 = feature_dim
    hidden_size2 = 250  # 200#250
    hidden_size3 = 100
    # feature_dim ——> hidden_size1=feature_dim*2
    # feature_dim*6+1 ——> hidden_size2 ——> hidden_size2 * 8(concat)
    # hidden_size2 * 8 ——> hidden_size3 ——> num_classes=34
    model = RNN(feature_dim, hidden_size1, hidden_size2, hidden_size3, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model