from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision.transforms import Compose
from torch.optim import Adam
from utils import *
from models import *

epochs = 1000
batch_size = 128
lr = 0.001

transform_train = Compose(
    [
    ]
)

transform_val = Compose(
    [
    ]
)


model = CnnLstm()
optimizer = Adam(model.parameters(), lr=lr)

train_data = shrec19(
    '/home/ken/Downloads/shrec2019/output/ring0/', train=True)
val_data = shrec19(
    '/home/ken/Downloads/shrec2019/output/ring0/', train=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# train
for epoch in range(epochs):
    pass
