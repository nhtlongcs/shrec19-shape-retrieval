import sys
import os
import numpy as np
import glob
import math
# from ringview import take_a_snap

test_path = ''
train_path = ''
output_path = ''


class ShrecDataset(object):
    """Some Information about dataset"""

    def __init__(self, root, train=True, filter_empty=True):
        super().__init__()
        self.root = root
        test_path = root + 'list/model_train.txt'
        train_path = root + 'list/model_train.txt'
        output_path = root + 'output/'
        self.list = []

        self.is_train = train

        if self.is_train:
            f = open(train_path, "r")
        else:
            f = open(test_path, "r")

        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            self.list.append((line[0], int(line[1])))
        f.close()

    def __getitem__(self, index):
        obj = self.root + self.list[index][0]

        if self.is_train:
            label = self.list[index][1]
            # print('Imported name: {} \nClass: {}'.format(obj,
            #                                             label))
            return (obj, label)
        else:
            # print('Imported name: {} '.format(obj))
            return (obj)

    def __len__(self):
        return len(self.list)

    def export_2d(self):
        # for i in range(len(self)):
        for i in range(5):
            obj_path = self.root + 'data/' + (self[i][0])
            output_path = str(self[i][1]) + '/' + \
                self[i][0].split('/')[1].split('.')[0] + '/'
            #
            print(obj_path)
            print(output_path)

            cnt = 0

            for i in [4, -4]:

                z = math.sin(math.pi/i)
                y = math.cos(math.pi/i)
                print(self.root + 'output/ring' + str(cnt) + '/' + output_path)
                # take_a_snap(obj_path,
                #             self.root + 'output/ring' + str(cnt) + '/' + output_path, (0, y, z))
                cnt += 1

            print(self.root + 'output/ring' + str(cnt) + '/' + output_path)
            # take_a_snap(obj_path,
            #             self.root + 'output/ring' + str(cnt) + '/' + output_path, (0, -1, 0))


if __name__ == "__main__":
    dataset = ShrecDataset(
        root="/home/ken/Downloads/shrec2020-data-supervise/supervise/")
    print(dataset[0])
    # dataset.export_2d()
