from .RNN_LSTM_V1 import lstm_total_model
from .RNN_LSTM_V2 import lstm_channel8_model
from .RNN_GRU_V1 import gru_channel8_model
from .RNN_GRU_V2_1 import gru_channel8_attention_model
from .RNN_GRU_V2_2 import rnn_attention_gru_model
from .RNN_GRU_V3 import gru_channel8_attention_age_gender_model

from .SERestNet_gru_model_V1 import SERestNet_gru_model
from .SERestNet_gru_model_V2 import SERestNet_gru_channel8_model
from .SERestNet_gru_model_V3 import SERestNet_gru_sep8_model

from .ResNet import ResNet34, ResNet50, ResNet101, ResNet152
from .ResNext import ResNeXt50_2x16d, ResNeXt50_2x32d, ResNeXt50_2x64d, ResNeXt50_4x64d
from .ResNext import ResNeXt101_2x64d, ResNeXt101_4x64d
from .ResNext import ResNeXt152_2x64d, ResNeXt152_4x64d
from .ResNet_Basic import ResNet50_Basic, ResNet101_Basic
from .ResNext_Basic import ResNeXt50_2x64d_Basic, ResNeXt101_4x64d_Basic
from .SEResNet import SEResNet50, SEResNet101
from .ECGNet_V1 import ECGNet_v1
from .ECGNet_V2 import ECGNet_v2
from .ECGNet_V3 import ECGNet_v3