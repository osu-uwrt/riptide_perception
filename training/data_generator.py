import bpy
import os
import random
import mathutils
from pathlib import Path

def randomizeScene(obj_name):
    #Get particle system named Particle Area in main collection
    particles = bpy.data.collections['Main'].objects['Particle Area'].particle_systems[0].particles
    #change seed to random number
    particles.seed = random.randint(0, 100000)
    #Get object obj_name
    obj = bpy.data.objects[obj_name]
    #Set object's rotation/position to be within a 2x2 sphere
    while True:
        pos = mathutils.Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
        if pos.length <= 1:
            break
    obj.location = pos * 2
    obj.rotation_euler = mathutils.Euler((random.uniform(0, math.pi*2), random.uniform(0, math.pi*2), random.uniform(0, math.pi*2)), 'XYZ')
    #TODO: Check to see if the object is visible enough, redo if not.



def main():
    import sys       # to get command line args
    import argparse  # to parse options for us and print a nice help message
    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments
    argv = sys.argv
    bpy.ops.wm.open_mainfile(filepath="./UWRTBase.blend")
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # When --help or no args are given, print this help
    usage_text = (
        "Generate YOLOv5 training data with this script:"
        "  blender --python --background " + __file__ + " -o [output_file] -t [training_obj]"
    )

    parser = argparse.ArgumentParser(description=usage_text)

    training_collection = bpy.data.collections['Training Objects']
    training_names = []
    training_names_helper = "\n"
    for training_object in training_collection.objects[:]:
        training_names.append(training_object.name)
        training_names_helper += " "+training_object.name+","
    parser.add_argument(
        "-t", "--train", dest="training_obj", metavar='FILE',
        help=(
        "Name of the object you wan't to generate data for."
        "\n Supported names: " + training_names_helper
        ),
    )
    parser.add_argument(
        "-o", "--output", dest="output_path", metavar='FILE',
        help="Location to store generated files",
    )
    parser.add_argument(
        "-n", "--number", dest="number_datapoints", type=int,
        help="Location to store generated files", default=1,
    )

    args = parser.parse_args(argv)  # In this example we won't use the args

    if not argv:
        parser.print_help()
        return

    if training_obj not in training_names:
        print("Object name "+training_obj+" not trainable, try "+training_names_helper)
        return
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    img_output_dir = output_path.joinpath('images/')
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir = output_path.joinpath('labels/')
    label_output_dir.mkdir(parents=True, exist_ok=True)
    bounding_box_file = open(output_path+"/image_bounding_boxes.txt","w")
    # Run the example function
    for i in range(number_datapoints):
        randomlyPlaceObject(training_obj)
        
        #Get output paths for img and data
        img_output = img_output_dir / f"im{i}.jpg"        
        label_output = label_output_dir / f"im{i}.txt" 
        
        # Render image
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        
        # Get object's 2D image bounding box
        obj = bpy.data.objects[obj_name]
        bbox_2d = obj.bound_box[:]
        bbox_2d = [mathutils.Vector((bbox_2d[i][0], bbox_2d[i][1])) for i in range(8)]
        min_x = min([p.x for p in bbox_2d])
        max_x = max([p.x for p in bbox_2d])
        min_y = min([p.y for p in bbox_2d])
        max_y = max([p.y for p in bbox_2d])
        
        # Output 2D bounding box to text file
        with open(label_output, "w") as f:
            x_center = (min_x + max_x) / 2
            y_center = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            f.write(f"0 {x_center} {y_center} {width} {height}\n")       
        

    print("Generation finished, exiting")


if __name__ == "__main__":
    main()
