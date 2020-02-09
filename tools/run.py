import os
import sys
import bpy
import time
import glob
import math
import mathutils
import numpy as np
from mathutils import Vector

test_path = ''
train_path = ''
output_path = ''

main_scene = bpy.context.scene
delta = 0
cameraOrigin = None
camera = None


def log(index):
    f = open("save.txt", "w+")
    f.write("%d" % (index))
    f.close()


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
        lamp_data.energy = 0.25

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

    def new_camera(self, camera_location):
        cam = bpy.data.cameras.new("Camera")
        cam.lens = 18

        self.camera = bpy.data.objects.new("Camera", cam)
        self.camera.location = camera_location
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


def setRotate():
    print("start")
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(rotateCamera)
    bpy.ops.render.render(animation=True)


def take_a_snap(obj_path, output_path, camera_location=(0, -1, 0)):

    # check blender version == 2.79
    print(bpy.app.version_string)
    scene = Scene(render_path=output_path+'/render/Image####.png')

    scene.add_to_scene(obj_path)

    scene.normalize()
    for z in [-1, 1]:
        for x in range(2):
            for y in range(2):
                location = (pow(-1, x), pow(-1, y), z)
                scene.add_lighting(location)

    scene.new_camera(camera_location)
    global delta
    delta = scene.delta
    global camera
    global cameraOrigin
    camera = bpy.data.objects['Camera']
    cameraOrigin = np.array(camera.location)

    # depth mode

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.size = [0.7]
    map.use_min = True
    map.min = [0]
    map.use_max = True
    map.max = [1]
    links.new(rl.outputs[2], map.inputs[0])

    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])

    # The viewer can come in handy for inspecting the results in the GUI
    depthViewer = tree.nodes.new(type="CompositorNodeViewer")
    links.new(invert.outputs[0], depthViewer.inputs[0])
    # Use alpha from input.
    links.new(rl.outputs[1], depthViewer.inputs[1])

    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = output_path + 'depth/'
    links.new(invert.outputs[0], fileOutput.inputs[0])
    ##########################################################

    # mask mode

    # clear default nodes

    # create input render layer node
    rl2 = tree.nodes.new('CompositorNodeRLayers')

    map2 = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map2.size = [0.00001]
    map2.use_min = True
    map2.min = [0]
    map2.use_max = True
    map2.max = [255]
    links.new(rl2.outputs[2], map2.inputs[0])

    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map2.outputs[0], invert.inputs[1])

    # The viewer can come in handy for inspecting the results in the GUI
    maskViewer = tree.nodes.new(type="CompositorNodeViewer")
    links.new(invert.outputs[0], maskViewer.inputs[0])
    # Use alpha from input.
    links.new(rl2.outputs[1], maskViewer.inputs[1])

    maskOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    maskOutput.base_path = output_path + 'mask/'
    links.new(invert.outputs[0], maskOutput.inputs[0])

    setRotate()


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
        f = open("save.txt", "r")
        start = int(f.read())
        for id in range(start, len(self)):
            # for i in range(1):
            obj_path = self.root + (self[id][0])
            output_path = str(self[id][1]) + '/' + \
                self[id][0].split('/')[1].split('.')[0] + '/'

            # print(obj_path)
            # print(output_path)

            cnt = 0

            for i in [4, -4]:
                for k in range(1, 5):
                    z = math.sin(math.pi/i*k)
                    y = math.cos(math.pi/i*k)
                    take_a_snap(obj_path,
                                self.root + 'output/ring' + str(cnt) + '/' + output_path, (0, y, z))
                    cnt += 1

            # take_a_snap(obj_path,
            #             self.root + 'output/ring' + str(cnt) + '/' + output_path, (0, -1, 0))
            log(id)


if __name__ == "__main__":
    # Get the scene
    scene = bpy.data.scenes["Scene"]

    # Set render resolution
    scene.render.resolution_x = 224
    scene.render.resolution_y = 244
    scene.render.resolution_percentage = 100

    dataset = ShrecDataset(
        root="/home/ken/Downloads/shrec2020-data-supervise/supervise/")
    dataset.export_2d()
