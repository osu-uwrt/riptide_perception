BLENDER DATA GENERATOR

Inside the UWRTBase.blend file there are three collections:

[1] Main - Do not touch
[2] Distractors - Everything in this collection will be shrunk and randomly placed as distractor objects
[3] Training Objects - These objects and their names correspond to the objects you want to generate data for.

To run:
```
blender --background .\UWRTBase.blend --python ./data_generator.py --  -o ./lol -t TrainObj1 -n 10
```

make sure that blender is on the path, and that the console is in this directory.

