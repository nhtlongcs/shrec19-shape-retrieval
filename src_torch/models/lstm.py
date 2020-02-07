from torch import nn
from torch import cat, randn
import torch.nn.functional as F
import torch
__all__ = ['Lstm']


class Lstm(nn.Module):
    """Some Information about Lstm"""

    def __init__(self, embed_size=2048, units=128):
        super(Lstm, self).__init__()

        self.lstm = nn.LSTM(embed_size, units,
                            bidirectional=True, batch_first=True)

    def forward(self, x):

        h_lstm, _ = self.lstm(x)
        # global average pooling
        # avg_pool = torch.mean(h_lstm, 1)  # print(avg_pool.shape)

        return h_lstm


if __name__ == "__main__":
    model = Lstm()
    model(torch.rand(2, 3, 2048))
