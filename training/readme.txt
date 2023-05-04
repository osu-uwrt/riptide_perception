BLENDER DATA GENERATOR

Inside the UWRTBase.blend file there are three collections:

[1] Main - Do not touch
[2] Distractors - Everything in this collection will be shrunk and randomly placed as distractor objects
[3] Training Objects - These objects and their names correspond to the objects you want to generate data for.

To run:
```
python3 data_generator.py -n {number of datapoints} -o {output path} -t {name of test object}
```

(Note probably are bugs as I wrote the code without having bpy installed due to literally having megabytes of storage left on my pc, patches to come)
