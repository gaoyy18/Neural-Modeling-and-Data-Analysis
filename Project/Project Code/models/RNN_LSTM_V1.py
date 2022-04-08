import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['lstm_total_model']

class LSTM0(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2):
        super(LSTM0, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_szie1, 2, dropout=0.3, bidirectional=True)
        self.r2o = nn.Linear(hidden_szie1 * 2, hidden_szie2)
        self.T = 5000

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

        self.rnn=LSTM0(input_size, hidden_size1, hidden_size2)
        self.batch = 128
        self.mlp = MLP(hidden_size2, hidden_size3, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2).clone()

        rnn_out = self.rnn(x.transpose(0,1))
        output = self.mlp(rnn_out)
        return output


def lstm_total_model(pretrained=False, **kwargs):
    """Constructs a rnn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    feature_dim = 8
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