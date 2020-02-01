from torch import nn
from torch import cat, randn
from .resnet import ResNet
import torch.nn.functional as F
import torch
__all__ = ['CnnLstm']


class CnnLstm(nn.Module):
    """Some Information about CnnLstm"""

    def __init__(self, total_classes=21):
        super(CnnLstm, self).__init__()
        embed_size = 24576  # 2048 x 12

        self.embedding = ResNet()

        self.lstm1 = nn.LSTM(embed_size, 128,
                             bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(128 * 2, 128,
                             bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(128 * 4, 128 * 4)
        self.linear2 = nn.Linear(128 * 4, 128 * 4)

        self.linear_out = nn.Linear(128 * 4, 1)
        self.linear_aux_out = nn.Linear(128 * 4, total_classes)

    def forward(self, x):
        tmp = []
        # print(list(x.shape)[1])

        for i in range((list(x.shape)[1])):
            tmp.append(self.embedding(x[:, i]))  # [1, embed_size]
        tmp = cat(tmp, dim=1).unsqueeze(1)  # [B, 1 , 12 x embed_size]

        h_lstm1, _ = self.lstm1(tmp)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], dim=1)

        return out
