# Implementation

## 1. System Overview

The project implements an object detection pipeline for construction-site
helmet and PPE detection using YOLO.

The pipeline consists of:

1. Dataset preparation
2. Model training
3. Model evaluation
4. Image/video inference
5. Result visualization

## 2. Model

YOLO is used as the object detection framework.

The project evaluates a baseline model and an improved model using the
same test dataset.

## 3. Evaluation Pipeline

The evaluation module calculates detection performance metrics including:

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95

The evaluation scripts are located in:

`src/evaluation/`

## 4. Inference Pipeline

The inference module accepts an image, directory of images, or video
source and generates predicted bounding boxes and class labels.

The inference implementation is located in:

`src/inference/predict.py`

## 5. Demo

A standalone demonstration script is provided at:

`src/demo/demo.py`

Example:

```bash
python src/demo/demo.py --model models/best.pt --source demo/sample.jpg