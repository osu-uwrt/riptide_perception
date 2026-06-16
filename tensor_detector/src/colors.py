#!/usr/bin/env python3

# Colors used for rviz markers, keyed by the published class name.
# Will move to yaml eventually
COLOR_MAP = {
    # Slaloms (slalom_name set by service)
    'slalom_close': (1.0, 0.0, 0.0),
    'slalom_middle': (1.0, 1.0, 0.0),
    'slalom_far': (0.0, 1.0, 0.0),

    # Slalom publishes under slalom_name, this is the default (not sure if we are using the service anymore)
    'slalom_front': (0.0, 1.0, 0.4),

    # Gate combinations
    'gate_repair': (0.0, 1.0, 0.0),
    'gate_rescue': (0.0, 0.5, 1.0),

    # Torpedo stuff 
    'torpedo': (0.0, 1.0, 1.0),
    'fire_hole_large': (1.0, 0.4, 0.0),
    'fire_hole_small': (1.0, 0.7, 0.0),
    'blood_hole_large': (0.6, 0.0, 0.0),
    'blood_hole_small': (1.0, 0.0, 0.4),

    # FFC generic planar classes (when not merged into a gate pair)
    'buoy': (0.2, 0.8, 1.0),
    'compass': (0.9, 0.6, 0.1),
    'hammer_and_wrench': (0.4, 0.4, 0.45),
    'sos': (1.0, 0.1, 0.3),

    # DFC generic planar classes
    # fire/blood also fall here on dfc, as the torpedo task is disabled
    'bandage': (0.95, 0.85, 0.70),
    'blood': (0.80, 0.0, 0.0),
    'fire': (1.0, 0.5, 0.0),
    'helmet': (1.0, 1.0, 0.0),
    'nut_and_bolt': (0.5, 0.5, 0.5),
    'pill': (0.7, 0.2, 0.9),
    'plug': (0.0, 0.4, 1.0),
    'warning': (1.0, 0.0, 1.0),
}
