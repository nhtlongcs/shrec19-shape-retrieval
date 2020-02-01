import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tf


__all__ = ['shrec19']


class shrec19(torch.utils.data.Dataset):
    """Some Information about shrec19"""

    def __init__(self, root: str, train=True, classes=20, DummyTransform=tf.Compose([tf.ToTensor()])):
        super(shrec19, self).__init__()

        self.is_train = train

        # if self.is_train:
        #     self.image_path = root + 'train/'
        # else:
        #     self.image_path = root + 'val/'

        self.image_path = root
        self.extension = '.png'
        self.transform = DummyTransform

        self.depth_folder = 'depth/'
        self.alpha_folder = 'mask/'
        self.render_folder = 'render/'

        self.list_obj = ([(os.path.splitext(os.path.basename(
            x))[0], idx) for idx in range(classes+1) for x in (os.listdir(self.image_path + str(idx) + '/'))])

    def __getitem__(self, index):
        data = []

        label = self.list_obj[index][1]
        obj_path = self.image_path + str(label) + '/' + self.list_obj[index][0]

        # for folder in [self.depth_folder, self.alpha_folder, self.render_folder]: -> new f*cking dims :?
        for folder in [self.depth_folder]:

            for idx in range(1, 13):

                path = obj_path + '/' + folder + 'Image' + \
                    "{:04d}".format(idx) + self.extension
                im = Image.open(path).convert('RGB')
                # [C, W, H]
                im = self.transform(im).unsqueeze(0)  # -> [B, C, W, H] B = 1

                data.append(im)

        data = torch.cat(data, dim=0)  # -> [B, C, W, H] B = 12
        return data, label

    def __len__(self):
        return len(self.list_obj)


if __name__ == "__main__":
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
    for im, label in dataset:
        break
