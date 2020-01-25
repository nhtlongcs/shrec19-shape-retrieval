import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tf


__all__ = ['shrec19']


class shrec19(torch.utils.data.Dataset):
    """Some Information about shrec19"""

    def __init__(self, root: str, train=True, classes=20):
        super(shrec19, self).__init__()

        self.is_train = train

        # if self.is_train:
        #     self.image_path = root + 'train/'
        # else:
        #     self.image_path = root + 'val/'

        self.image_path = root

        self.list_obj = ([(os.path.splitext(os.path.basename(
            x))[0], idx) for idx in range(classes+1) for x in (os.listdir(self.image_path + str(idx) + '/'))])

        print(len(self.list_obj))
        self.scale = 1

    def __getitem__(self, index):
        data = []
        label = self.list_obj[index][1]
        obj_path = self.image_path + str(label) + '/' + self.list_obj[index][0]

        for idx in range(1, 12):

            im = Image.open(obj_path + '/' + "{:04d}".format(idx) + '.png')
            im = self.preprocess(im, self.scale)
            im = tf.ToTensor()(im)

            data.append(im)

        # assert len(data)==12
        return data, label

    def __len__(self):
        return len(self.list_obj)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans


# if __name__ == "__main__":
#     dataset = shrec19(
#         '/home/ken/Downloads/shrec2019/output/ring0/', train=True)
#     for im, label in dataset:
#         print(label)
