# Construction Site Safety Dataset Report

## 1. Dataset Source

- **Dataset Name:** Construction Site Safety
- **Source:** Roboflow Universe
- **Version:** 30
- **License:** CC BY 4.0
- **Annotation Format:** YOLOv8

Dataset URL:
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

---

## 2. Dataset Statistics

| Split | Number of Images |
|--------|-----------------:|
| Training | 521 |
| Validation | 114 |
| Testing | 82 |
| **Total** | **717** |

Number of Classes: **25**

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

```
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

The dataset uses the YOLOv8 annotation format. Each image has a corresponding `.txt` annotation file containing the class ID and normalized bounding-box coordinates.

---

## 6. Dataset Verification

The dataset was successfully verified using Ultralytics YOLOv8.

Verification completed:

- Dataset loaded successfully.
- Folder structure verified.
- `data.yaml` configuration verified.
- Training, validation, and testing folders detected correctly.
- Random training images were inspected.
- Bounding boxes aligned correctly with the annotated objects.

---

## 7. Initial Observations

- Images represent real construction-site environments.
- Dataset includes workers, PPE equipment, machinery, and vehicles.
- Images contain varying lighting conditions.
- Some workers are partially occluded.
- Object sizes vary from small to large.
- PPE-related classes include Hardhat, Safety Vest, Mask, NO-Hardhat, NO-Safety Vest, and NO-Mask.

---

## 8. Planned Preprocessing

- Resize images to **640 × 640**.
- Preserve YOLOv8 annotation format.
- Normalize images during training.
- Use the provided train, validation, and test split.

---

## 9. Planned Data Augmentation

- Horizontal flipping
- Mosaic augmentation
- Random scaling
- HSV color augmentation
- Brightness variation

---

## 10. Conclusion

The Construction Site Safety dataset has been successfully downloaded, verified, and prepared for model training. The dataset is suitable for developing a YOLOv8-based PPE detection system and is ready to be used by Member 2 for baseline model training.