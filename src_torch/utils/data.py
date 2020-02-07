import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tf
import matplotlib.pyplot as plt

__all__ = ['shrec19']


class shrec19(torch.utils.data.Dataset):
    """Some Information about shrec19"""

    def __init__(self, root: str, Len=13, rings=8, train=True, classes=20, DummyTransform=tf.Compose([tf.ToTensor()])):
        super(shrec19, self).__init__()

        self.is_train = train

        # if self.is_train:
        #     self.image_path = root + 'train/'
        # else:
        #     self.image_path = root + 'val/'
        self.rings = rings
        self.len = Len
        self.image_path = root
        self.extension = '.png'
        self.transform = DummyTransform

        self.depth_folder = 'depth/'
        self.alpha_folder = 'mask/'
        self.render_folder = 'render/'

        # Get list obj name + classes ex : /home/ken/Downloads/shrec2019/output/ring0/0/
        self.list_obj = ([(os.path.splitext(os.path.basename(
            x))[0], idx) for idx in range(classes+1) for x in (os.listdir(self.image_path + '0/' + str(idx) + '/'))])

    def __getitem__(self, index):

        data = []
        label = self.list_obj[index][1]
        for ring in range(self.rings):
            obj_path = self.image_path + str(ring) + '/' + \
                str(label) + '/' + \
                self.list_obj[index][0]  # example : /home/ken/Downloads/shrec2019/output/ring2/0/12gh4i3fe
            ring_packed = []
            # for folder in [self.depth_folder, self.alpha_folder, self.render_folder]: -> new f*cking dims :?
            for folder in [self.depth_folder]:

                for idx in range(1, self.len):  # for with length

                    path = obj_path + '/' + folder + 'Image' + \
                        "{:04d}".format(idx) + self.extension
                    # example : /home/ken/Downloads/shrec2019/output/ring7/0/12gh4i3fe/depth/Image0010.png
                    im = Image.open(path).convert('RGB')
                    # [C, W, H]
                    im = self.transform(im).unsqueeze(
                        0)  # -> [L, C, W, H] B = 1

                    ring_packed.append(im)
                # -> [L, C, W, H] L = 12
                ring_packed = torch.cat(ring_packed, dim=0).unsqueeze(0)
            data.append(ring_packed)

        # -> [R, L, C, W, H] L = 12 ( R : ring )
        data = torch.cat(data, dim=0)
        # print(data.shape) torch.Size([8, 12, 3, 224, 224])

        # plt.imshow(data[0][0].numpy().transpose(1, 2, 0))
        # plt.show()
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
        '/home/ken/Downloads/shrec2019/output/ring', train=True, DummyTransform=dummytf)
    for im, label in dataset:
        break
