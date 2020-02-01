from torch import nn
from torch import cat, randn
from .resnet import ResNet

__all__ = ['CnnLstm']


class CnnLstm(nn.Module):
    """Some Information about CnnLstm"""

    def __init__(self):
        super(CnnLstm, self).__init__()
        self.extract_feature = ResNet()

    def forward(self, x):
        tmp = []
        for i in range(list(x.shape)[0]):
            tmp.append(self.extract_feature(x[i].unsqueeze(0)))  # [1, 2048]
        tmp = cat(tmp, dim=1)

        return tmp  # [1, 12 x 2048]
