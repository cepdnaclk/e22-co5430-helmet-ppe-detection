# Construction Site Safety Dataset Report

## 1. Dataset Source

* **Dataset Name:** Construction Site Safety
* **Source:** Roboflow Universe
* **Version:** 30
* **License:** CC BY 4.0
* **Annotation Format:** YOLOv8

Dataset URL:
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

---

## 2. Dataset Statistics

| Split      |  Images | Label Files |
| ---------- | ------: | ----------: |
| Training   |     521 |         521 |
| Validation |     114 |         114 |
| Testing    |      82 |          82 |
| **Total**  | **717** |     **717** |

* **Number of classes:** 25
* **Total annotated objects:** 5,570

The dataset is divided into separate training, validation, and testing sets. The corresponding number of image and label files is consistent across all three splits.

---

## 3. Classes

1. Excavator
2. Gloves
3. Hardhat
4. Ladder
5. Mask
6. NO-Hardhat
7. NO-Mask
8. NO-Safety Vest
9. Person
10. SUV
11. Safety Cone
12. Safety Vest
13. bus
14. dump truck
15. fire hydrant
16. machinery
17. mini-van
18. sedan
19. semi
20. trailer
21. truck and trailer
22. truck
23. van
24. vehicle
25. wheel loader

---

## 4. Dataset Structure

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 5. Annotation Format

The dataset uses the YOLO annotation format.

Each annotation contains five values:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized to the range 0 to 1.

---

## 6. Annotation Validation

A custom Python validation script was developed to check the dataset annotations.

The validation checks:

* Correct YOLO annotation structure
* Five values per annotation
* Valid class IDs from 0 to 24
* Normalized bounding-box coordinates
* Bounding-box values within the range 0 to 1

### Validation Results

* **Label files checked:** 717
* **Annotations checked:** 5,570
* **Errors found:** 0

Therefore, all checked YOLO annotations satisfy the required format and value constraints.

The validation script is available at:

```text
src/preprocessing/validate_annotations.py
```

---

## 7. Class Distribution Analysis

A class distribution analysis was performed across the training, validation, and testing annotations.

The results show that the dataset is imbalanced across the 25 classes.

The most frequent classes include:

| Class          | Annotations |
| -------------- | ----------: |
| Person         |       1,148 |
| Safety Cone    |         600 |
| NO-Safety Vest |         582 |
| Hardhat        |         574 |
| NO-Mask        |         491 |

Some classes have very few examples:

| Class        | Annotations |
| ------------ | ----------: |
| bus          |           1 |
| fire hydrant |           6 |
| mini-van     |           7 |
| semi         |           7 |
| truck        |           7 |

This class imbalance should be considered when interpreting model performance, particularly for classes with very few training examples.

The generated class distribution chart is available at:

```text
results/plots/class_distribution.png
```

The analysis script is available at:

```text
src/preprocessing/class_distribution.py
```

---

## 8. Initial Dataset Observations

The dataset represents real construction-site environments and contains workers, PPE equipment, machinery, and vehicles.

Important characteristics include:

* Different lighting conditions
* Partial object occlusion
* Different object sizes
* Crowded construction-site scenes
* PPE-related classes such as Hardhat, Safety Vest, Mask, NO-Hardhat, NO-Safety Vest, and NO-Mask

These characteristics provide realistic challenges for object detection.

---

## 9. Preprocessing

The YOLO training pipeline uses an image size of **640 × 640 pixels**.

The original train, validation, and test splits are maintained separately to support fair model evaluation and avoid data leakage.

The YOLO annotation format is preserved throughout the dataset preparation process.

---

## 10. Data Augmentation

Data augmentation is used during model training to improve robustness to variations commonly found in construction-site images.

The planned augmentation techniques include:

* Horizontal flipping
* Mosaic augmentation
* Random scaling
* HSV/color augmentation
* Brightness variation
* Translation

These transformations are intended to improve robustness to changes in lighting, viewpoint, object position, scale, and scene composition.

Augmentation should be applied to the training data only. Validation and test data should remain unchanged for fair evaluation.

---

## 11. Dataset Quality Summary

The completed dataset analysis produced the following results:

* **717 images**
* **717 corresponding label files**
* **5,570 annotations**
* **25 classes**
* **0 annotation validation errors**
* **Class distribution analysis completed**
* **Class imbalance identified**

The dataset is therefore structurally suitable for the subsequent model training and evaluation stages.

---

## 12. Conclusion

The Construction Site Safety dataset has been downloaded, structured, and validated for the YOLO-based PPE detection project.

The annotation validation confirmed that all 717 label files and 5,570 annotations satisfy the required YOLO format and coordinate constraints. Class distribution analysis also identified significant differences in the number of examples between classes.

These dataset characteristics will be considered during the model training and evaluation stages.
