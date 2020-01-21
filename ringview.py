import bpy
import numpy as np
import math
import mathutils
import time
from mathutils import Vector

main_scene = bpy.context.scene
delta = 0
cameraOrigin
camera


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


def setRotate():
    print("start")
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(rotateCamera)
    bpy.ops.render.render(animation=True)


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


# take_a_snap('/home/ken/Downloads/shrec2019/data/3D_OBJ/3i4gf2he1.obj',
#             '/home/ken/scripts blender/test/1/')
