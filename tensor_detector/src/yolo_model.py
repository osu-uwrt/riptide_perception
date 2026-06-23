#!/usr/bin/env python3
"""Thin wrapper around the Ultralytics YOLO model"""
import os
import torch
from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path, export=False, logger=None):
        self.model = None
        self._export = export
        self._logger = logger
        self._initialize(model_path)

    def _warn(self, msg):
        if self._logger:
            self._logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    def _initialize(self, model_path):
        # Prefer a prebuilt .engine next to the .pt if one exists.
        # only if CUDA is available
        engine_model_path = os.path.splitext(model_path)[0] + '.engine'
        if model_path.endswith(".pt") and os.path.exists(engine_model_path):
            if torch.cuda.is_available():
                model_path = engine_model_path
            else:
                self._warn(f"CUDA unavailable, using PT model instead of engine: {model_path}")
        self.model = YOLO(model_path, task="segment")

        # Optionally export to TensorRT, then reload the freshly built engine.
        if self._export and model_path.endswith(".pt"):
            if torch.cuda.is_available():
                self.model.export(format="engine")
                self._initialize(engine_model_path)
            else:
                self._warn("export=True requested but CUDA is unavailable, skipping TensorRT export.")

    def infer(self, image, conf, iou):
        return self.model(image, verbose=False, iou=iou, conf=conf)
