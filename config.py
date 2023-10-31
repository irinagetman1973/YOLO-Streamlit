#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Irina Getman
   @Date:          1/09/2023
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys
import torch


# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video"]


# DL model config
DETECTION_MODEL_DIR_V8= ROOT / 'weights' / 'detection'
YOLOv8l = DETECTION_MODEL_DIR_V8 / "yolov8l.pt"
YOLOv8m = DETECTION_MODEL_DIR_V8 / "yolov8m.pt"
YOLOv8n = DETECTION_MODEL_DIR_V8 / "yolov8n.pt"
# YOLOv8s = DETECTION_MODEL_DIR_V8 / "yolov8s.pt"
YOLOv8x = DETECTION_MODEL_DIR_V8 / "yolov8x.pt"
DriveWatch = DETECTION_MODEL_DIR_V8 / "DriveWatch.pt"
YOLOv8_Champion = DETECTION_MODEL_DIR_V8 / "v8_champion.pt"

# DL model config for YOLOv7
DETECTION_MODEL_DIR_V7 = ROOT / 'weights' / 'detection' 
YOLOv7 = DETECTION_MODEL_DIR_V7 / "yolov7.pt"
YOLOv7_Champion = DETECTION_MODEL_DIR_V8 / "v7_champion.pt"
YOLOv7_e6 = DETECTION_MODEL_DIR_V7 / "yolov7-e6.pt"
YOLOv7_w6 = DETECTION_MODEL_DIR_V7 / "yolov7-w6.pt"
YOLOv7x = DETECTION_MODEL_DIR_V7 / "yolov7x.pt"

DETECTION_MODEL_LIST_V8 = [
    "yolov8l.pt",
    "yolov8m.pt",
    "yolov8n.pt",
    "yolov8x.pt",
    "DriveWatch.pt",
    "v8_champion.pt",
    
    ]
DETECTION_MODEL_LIST_V7 = [
    "yolov7.pt",
    "v7_champion.pt",
    "yolo7-e6.pt",
    "yolov7-w6.pt",
    "yolov7x.pt"
]

OBJECT_COUNTER = None
OBJECT_COUNTER1 = None