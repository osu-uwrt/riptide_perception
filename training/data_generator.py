import bpy
import math
import os
import random
import mathutils
from pathlib import Path


def randomizeScene(possible_objects):
    # Get particle system named Particle Area in main collection
    particles = (
        bpy.data.collections["Main"]
        .objects["Particle Area"]
        .particle_systems[0]
        .particles
    )
    # change seed to random number
    # particles.seed = random.randint(0, 100000)
    # Decide which objects to use for this generation
    num_objects = random.randint(1, len(possible_objects))
    used_object_names = random.sample(possible_objects, num_objects)

    # Set each selected object's rotation/position to be within a 2x2 sphere
    for obj_name in used_object_names:
        obj = bpy.data.objects[obj_name]
        while True:
            pos = mathutils.Vector(
                (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            )
            if pos.length <= 1:
                break
        obj.location = pos * 2
        obj.rotation_euler = mathutils.Euler(
            (
                random.uniform(0, math.pi * 2),
                random.uniform(0, math.pi * 2),
                random.uniform(0, math.pi * 2),
            ),
            "XYZ",
        )


# END randomizeScene()


def parse_args(possible_training_names: str):
    import argparse  # to parse options for us and print a nice help message
    import sys

    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments

    # When --help or no args are given, print this help
    usage_text = (
        "Generate YOLOv5 training data with this script:"
        "  blender --python --background "
        + __file__
        + "-- -o [output_file] -t [training_obj1 [training_obj2 [...]]] -n [number_datapoints]"
    )
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument(
        "-t",
        "--train",
        dest="training_objs",
        metavar="TRAINING_OBJ",
        help=(
            "Names of objects you want to generate data for."
            "\n Supported names: " + possible_training_names
        ),
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        metavar="FILE",
        help="Location to store generated files",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--number",
        dest="number_datapoints",
        type=int,
        help="Number of datapoints to generate",
        default=1,
    )

    raw_args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    # Convert raw_args.training_objs to a dict numbering the objects in order
    # This is so we keep track of each object's ID in the training data
    training_objs = {}
    for i in range(len(raw_args.training_objs)):
        training_objs[raw_args.training_objs[i]] = i
    raw_args.training_objs = training_objs
    print(raw_args.training_objs)
    return raw_args


# END parse_args()


def print_bounding_boxes(label_output, visible_objects, object_dict):
    # Clear file
    with open(label_output, "w") as f:
        f.truncate(0)

    # Print the bounding box for each visible object
    for obj_name in visible_objects:
        obj = bpy.data.objects[obj_name]
        vision_index = object_dict[obj_name]

        # https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file/
        bbox_2d = camera_view_bounds_2d(
            bpy.context.scene, bpy.context.scene.camera, obj
        )

        r = bpy.context.scene.render
        # Output 2D bounding box to text file
        with open(label_output, "w") as f:
            f.append(
                f"{vision_index} {(bbox_2d.x + (bbox_2d.width/2))/r.resolution_x} {(bbox_2d.y+(bbox_2d.height/2))/r.resolution_y} {bbox_2d.width/r.resolution_x} {bbox_2d.height/r.resolution_y}\n"
            )


# END print_bounding_boxes()


def main():
    # Get names of all objects in the training collection for help message
    training_collection = bpy.data.collections["Training Objects"]
    training_names = []
    training_names_helper = "\n"
    for training_object in training_collection.objects[:]:
        training_names.append(training_object.name)
        training_names_helper += " " + training_object.name + ","

    args = parse_args(training_names_helper)

    # Check for incorrect args
    for obj in args.training_objs:
        if obj not in training_names:
            print(f"Object name {obj} not trainable, try {training_names_helper}")
            return

    output_path = Path(args.output_path)
    print(f"Generating to {output_path.resolve()}")
    output_path.mkdir(parents=True, exist_ok=True)
    img_output_dir = output_path / "images/"
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir = output_path / "labels/"
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each image
    for i in range(args.number_datapoints):
        used_objects = randomizeScene(args.training_objs)

        # Get output paths for img and data
        img_output = img_output_dir / f"im{i}.jpg"
        label_output = label_output_dir / f"im{i}.txt"

        # Render image
        bpy.context.scene.render.filepath = str(img_output.resolve())
        bpy.ops.render.render(write_still=True)

        # Get object's 2D image bounding box
        print_bounding_boxes(label_output, used_objects, args.training_objs)

    print("Generation finished, exiting")


class Box:
    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(scene, True, "PREVIEW")
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    # bpy.data.meshes.remove(me)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


if __name__ == "__main__":
    main()