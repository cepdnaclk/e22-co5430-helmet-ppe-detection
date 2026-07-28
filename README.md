# Construction Site PPE Compliance Detection using Deep Learning

## Project Overview

This project is developed for the CO543/CO5430 Computer Vision Project (2026).

The objective of this project is to automatically detect Personal Protective Equipment (PPE) compliance at construction sites using deep learning-based object detection techniques. The system identifies workers and determines whether they are wearing required safety equipment such as helmets and safety vests.

The project uses a YOLO-based object detection model trained on a publicly available construction site safety dataset.

---

## Problem Statement

Construction sites are high-risk working environments where workers are required to wear Personal Protective Equipment (PPE) such as helmets and safety vests to reduce workplace accidents.

Manual monitoring of PPE compliance is time-consuming, inconsistent, and difficult in large construction sites. This project aims to develop an automated Computer Vision solution capable of detecting PPE compliance from construction site images.

---

## Objectives

### Main Objective

Develop an object detection system capable of automatically detecting PPE compliance using deep learning.

### Specific Objectives

- Detect construction workers in images.
- Detect safety helmets and safety vests.
- Identify workers without required PPE.
- Evaluate detection performance using standard object detection metrics.
- Compare baseline and improved detection approaches.

---

## Dataset

**Dataset Name**

Construction Site Safety Dataset

**Source**

Roboflow Universe

**Project URL**

https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

**Annotation Format**

YOLO

**Classes**

- Person
- Helmet
- No Helmet
- Safety Vest
- No Safety Vest
- Mask
- Other PPE-related classes (depending on exported dataset version)

The dataset is publicly available and will only be used according to its license.

---

## Proposed Method

The project uses the YOLO object detection framework implemented using the Ultralytics library.

The overall workflow is:

1. Prepare the dataset.
2. Train a baseline YOLO model.
3. Fine-tune the model using the PPE dataset.
4. Evaluate model performance.
5. Compare baseline and improved approaches.
6. Perform failure-case analysis.

---

## Baseline Approach

The baseline model uses a pretrained YOLO detector with minimal or no fine-tuning.

This establishes a reference performance before applying project-specific improvements.

---

## Improved Approach

The improved model includes:

- Fine-tuning on the PPE dataset
- Data augmentation
- Hyperparameter tuning
- Performance comparison against the baseline model

---

## Evaluation Metrics

The following evaluation metrics will be used:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score
- Per-class performance

Qualitative evaluation will also be performed using prediction images.

---

## Repository Structure

```

dataset/
Raw dataset information and configuration files.

docs/
Project proposal, reports, presentations, and documentation.

models/
Trained model weights.

notebooks/
Training and experimentation notebooks.

results/
Prediction images, evaluation metrics, and plots.

src/
Python source code.

```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
```

Move into the project folder.

```bash
cd e22-co5430-helmet-ppe-detection
```

Install the required Python packages.

```bash
pip install -r requirements.txt
```

---

## Usage

Training and inference scripts will be added during project development.

---

## Results

Project results will include:

- Sample prediction images
- Performance metrics
- Confusion matrix
- Precision-Recall curves
- Loss curves

These outputs will be stored inside the `results/` directory.

---

## Team Contributions

| Member | Responsibility |
|--------|----------------|
| H.P.P.P.E. Umanda | Dataset preparation, preprocessing, annotation, evaluation |
| D.S. Wellage | Model implementation and training |
| R.V. Jayalath | Repository management, documentation, report integration and demo preparation |

---

## AI Usage Declaration

Artificial Intelligence tools were used for coding assistance, debugging support, documentation improvement, and writing clarity.

All generated content, code, and documentation were reviewed, verified, and understood by the project members before inclusion in the repository.