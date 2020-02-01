from models import *
from utils import *
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tf
from torch.optim import Adam
from torch import nn, manual_seed

n_epochs = 1000
batch_size = 128
lr = 0.001

if __name__ == "__main__":

    manual_seed(1)  # set random seed

    dummytf = tf.Compose(
        [
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )

    dataset = shrec19(
        '/home/ken/Downloads/shrec2019/output/ring0/', train=True, DummyTransform=dummytf)

    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len

    train_set, val_set = random_split(dataset, [train_len, val_len])
    print(len(train_set))
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = CnnLstm().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = Adam(model.parameters())

    outputs = model(train_set[0][0].to('cuda'))
    print(outputs.cpu().shape)
    for epoch in range(n_epochs):
        pass
