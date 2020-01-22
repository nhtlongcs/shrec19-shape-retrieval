import bpy
import numpy as np
import math
import mathutils
import time
from mathutils import Vector

import sys
import os
import glob

test_path = ''
train_path = ''
output_path = ''

main_scene = bpy.context.scene
delta = 0
cameraOrigin = None
camera = None


def reset_blend():

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


class Scene(object):
    def __init__(self, render_path, frame_end=12, render=False):
        super().__init__()
        # start a new scene
        reset_blend()
        # setup default render path
        bpy.data.scenes[0].render.filepath = render_path
        # setup totals render frames
        self.frame_end = frame_end
        bpy.context.scene.frame_end = self.frame_end

        # setup default params
        self.delta = 2 * math.pi / self.frame_end
        self.cnt = 0
        self.target = None  # main object
        self.camera = None  # main camera
        # setup render mode
        self.is_render = render

    def add_lighting(self, location):
        lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
        lamp_data.energy = 0.5

        # Create new object with our lamp datablock
        lamp_object = bpy.data.objects.new(
            name="New Lamp", object_data=lamp_data)

        # Link lamp object to the scene so it'll appear in this scene
        main_scene.objects.link(lamp_object)

        # Place lamp to a specified location
        lamp_object.location = location

        # And finally select it make active
        lamp_object.select = True
        main_scene.objects.active = lamp_object

    def add_to_scene(self, file_loc):
        bpy.ops.import_scene.obj(filepath=file_loc)
        self.target = bpy.context.selected_objects[0]  # <--Fix
        self.target

    def new_camera(self):
        cam = bpy.data.cameras.new("Camera")
        cam.lens = 18

        self.camera = bpy.data.objects.new("Camera", cam)
        self.camera.location = (0, -1, 0)
        self.camera.rotation_euler = (90, 0, 0)
        main_scene.camera = self.camera
        main_scene.objects.link(self.camera)

        targetobj = bpy.data.objects[self.target.name]
        pointyobj = bpy.data.objects['Camera']
        # make a new tracking constraint
        ttc = pointyobj.constraints.new(type='TRACK_TO')
        ttc.target = targetobj
        ttc.track_axis = 'TRACK_NEGATIVE_Z'
        ttc.up_axis = 'UP_Y'
        bpy.ops.object.select_all(action='DESELECT')

    def normalize(self):
        # calc ratio
        dimensions = list(self.target.dimensions)
        maxDimension = max(dimensions)
        ratio = 1/maxDimension
        # scale object to 1 blender unit
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (1, 0, 0))
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (0, 1, 0))
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (0, 0, 1))

        # set pivot object to center
        self.target.select = True
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        self.target.select = False

        # move object to world origin
        self.target.location = Vector((0, 0, 0))

    def clean(self):
        reset_blend()


def rotateCamera(scene):
    global delta
    angle = delta * scene.frame_current
    rotationMatrix = np.array([[math.cos(angle), math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0],
                               [0, 0, 1]])
    camera.location = np.dot(cameraOrigin, rotationMatrix)


def take_a_snap(obj_path, output_path):

    # check blender version == 2.79
    print(bpy.app.version_string)
    scene = Scene(render_path=output_path)

    scene.add_to_scene(obj_path)

    scene.normalize()
    for x in range(2):
        for y in range(2):
            location = (pow(-1, x), pow(-1, y), 1)
            scene.add_lighting(location)

    scene.new_camera()
    global delta
    delta = scene.delta
    global camera
    global cameraOrigin
    camera = bpy.data.objects['Camera']
    cameraOrigin = np.array(camera.location)

    setRotate()


def setRotate():
    print("start")
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(rotateCamera)
    bpy.ops.render.render(animation=True)


class ShrecDataset(object):
    """Some Information about dataset"""

    def __init__(self, root, train=True, filter_empty=True):
        super().__init__()
        self.root = root
        test_path = root + 'split/shrec2019_Model_test.txt'
        train_path = root + 'split/shrec2019_Model_train.txt'
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
        obj = self.list[index][0]

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
        for i in range(len(self)):
            # for i in range((5)):
            obj_path = self.root + 'data/' + (self[i][0])
            output_path = self.root + 'output/' + \
                str(self[i][1]) + '/' + \
                self[i][0].split('/')[1].split('.')[0] + '/'

            # print(obj_path)
            # print(output_path)
            take_a_snap(obj_path, output_path)


if __name__ == "__main__":
    dataset = ShrecDataset(root="/home/ken/Downloads/shrec2019/")
    dataset.export_2d()

# run code by cmd : blender -b -P '/home/ken/scripts blender/run.py'
