from models import *
from utils import *
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tf
from torch.optim import Adam
from torch import nn, manual_seed
import tqdm
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter


def parse_cmd_args():
    parser = argparse.ArgumentParser(
        description='run the procedure of advice opinion extraction')
    parser.add_argument('-p', '--path', type=str,
                        default='/home/ken/Downloads/shrec2019/output/ring0/', help='config file path')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='batchsize with each epochs')
    parser.add_argument('-n', '--n_epochs', type=int,
                        default=5, help='Total epochs')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.001, help='learning rate')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_cmd_args()
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    lr = config.lr
    dataset_path = config.path

    dummytf = tf.Compose(
        [
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )

    dataset = shrec19(dataset_path, train=True, DummyTransform=dummytf)

    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len

    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter()

    model = CnnLstm().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = Adam(model.parameters())
    count = 0
    with tqdm.tqdm(total=len(range(100)), file=sys.stdout) as pbar:
        for e in range(n_epochs):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                writer.add_scalar('Loss/train', loss,
                                  i + e*(train_len/batch_size))

                progress = float("{:.3f}".format(
                    batch_size/train_len/n_epochs*100.0))
                pbar.update(progress)

    print('Finished Training {}'.format(loss))
    # writer.close()

    torch.save(model.state_dict(), './baseline.pth')
