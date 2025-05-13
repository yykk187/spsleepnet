# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# --------------------- Utility ---------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# --------------------- Attention ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.fc_out(output)

        x = self.norm1(query + output)
        x = self.norm2(x + self.ffn(x))
        return x


# --------------------- Modules ---------------------


class Conv2dBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride[0] > 1 or stride[1] > 1:
            ConvLayer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=False, **kwargs)
        else:
            ConvLayer = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding='same', bias=False, **kwargs)
        super(Conv2dBnReLU, self).__init__(
            ConvLayer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.5, bidirectional=True, return_last=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.D = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        if self.return_last:
            out = out[:, -1, :]
        return out


# --------------------- DeepSleepNet ---------------------
class SPSleepNet(nn.Module):
    def __init__(self, n_timepoints, n_seqlen, n_classes, dropout=0.5,
                 n_filters_1=64, filter_size_1=50, filter_stride_1=6,
                 n_filters_2=64, filter_size_2=400, filter_stride_2=50,
                 pool_size_11=8, pool_stride_11=8,
                 pool_size_21=4, pool_stride_21=4,
                 n_filters_1x3=128, filter_size_1x3=8,
                 n_filters_2x3=128, filter_size_2x3=6,
                 pool_size_12=4, pool_stride_12=4,
                 pool_size_22=2, pool_stride_22=2,
                 n_rnn_layers=2, n_hidden_rnn=512, n_hidden_fc=1024,
                 n_heads=8):
        super().__init__()
        self.n_seqlen = n_seqlen

        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            nn.MaxPool2d((pool_size_11, 1), (pool_stride_11, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_12, 1), (pool_stride_12, 1)),
        )
        self.conv2 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_2, (filter_size_2, 1), (filter_stride_2, 1)),
            nn.MaxPool2d((pool_size_21, 1), (pool_stride_21, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_2, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            nn.MaxPool2d((pool_size_22, 1), (pool_stride_22, 1)),
        )
        self.drop1 = nn.Dropout(dropout)

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_11 // pool_stride_12
        outlen_conv2 = n_timepoints // filter_stride_2 // pool_stride_21 // pool_stride_22
        outlen_conv = outlen_conv1 * n_filters_1x3 + outlen_conv2 * n_filters_2x3

        self.res_branch = nn.Sequential(
            nn.Linear(outlen_conv, n_hidden_fc),
            nn.ReLU(inplace=True)
        )
        self.rnn_branch = LSTM(outlen_conv + 256, n_hidden_rnn, n_rnn_layers, bidirectional=True, return_last=False)
        self.drop2 = nn.Dropout(dropout)

        self.mha = MultiHeadAttention(n_heads, n_hidden_rnn * 2 + 256, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden_rnn * 2 + 256, n_classes),
            nn.LogSoftmax(dim=1)
        )
        self.position_fc = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(inplace=True)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, p):
        x = x.reshape((-1,) + x.shape[2:])  # [B*T, 1, time, 1]
        x1 = self.conv1(x).view(x.size(0), -1)
        x2 = self.conv2(x).view(x.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.drop1(x)

        x_seq = x.view(-1, self.n_seqlen, x.size(-1))
        # x_res = self.res_branch(x_seq[:, -1, :])

        p_features = self.position_fc(p)
        # p_res = p_features[:, -1, :]

        x_seq_combined = torch.cat((x_seq, p_features), dim=2)
        x_seq_rnn = self.rnn_branch.lstm(x_seq_combined)[0]

        x_attn_input = torch.cat((x_seq_rnn, p_features), dim=2)
        x_attn = self.mha(x_attn_input, x_attn_input, x_attn_input)
        x_attn_last = x_attn[:, -1, :]

        # x_combined = torch.cat((x_attn_last, p_res), dim=1)
        x = self.classifier(x_attn_last)
        return x


# --------------------- Test ---------------------
if __name__ == '__main__':
    x = torch.randn((16, 10, 1, 3000, 1))  # [batch, seq, 1, time, 1]
    p = torch.randn((16, 10, 5))           # posture features
    model = SPSleepNet(3000, 10, 5)
    y = model(x, p)
    print(y.shape)  # expected: [16, 5]