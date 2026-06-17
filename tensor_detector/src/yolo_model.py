#!/usr/bin/env python3
"""Thin wrapper around the Ultralytics YOLO model"""
import os
from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path, export=False):
        self.model = None
        self._export = export
        self._initialize(model_path)

    def _initialize(self, model_path):
        # Prefer a prebuilt .engine next to the .pt if one exists.
        engine_model_path = model_path.replace('.pt', '.engine')
        if model_path.endswith(".pt") and os.path.exists(engine_model_path):
            model_path = engine_model_path

        self.model = YOLO(model_path, task="segment")

        # Optionally export to TensorRT, then reload the freshly built engine.
        if self._export and model_path.endswith(".pt"):
            self.model.export(format="engine")
            self._initialize(engine_model_path)

    def infer(self, image, conf, iou):
        return self.model(image, verbose=False, iou=iou, conf=conf)
