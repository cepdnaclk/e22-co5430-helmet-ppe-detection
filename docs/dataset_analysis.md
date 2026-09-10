\# Dataset Analysis and Preprocessing



\## Dataset



Construction Site Safety Dataset - Version 30



\- Source: Roboflow Universe

\- Format: YOLOv8

\- License: CC BY 4.0

\- Number of classes: 25



\## Dataset Split



| Split | Images | Annotations |

|------|------:|------:|

| Train | 521 | 4031 |

| Validation | 114 | 733 |

| Test | 82 | 806 |

| \*\*Total\*\* | \*\*717\*\* | \*\*5570\*\* |



All image files have corresponding YOLO label files. No missing image-label pairs were detected.



\## Class Distribution



The dataset contains 25 object classes.



The most represented classes include:



\- Person: 1148

\- Safety Cone: 600

\- NO-Safety Vest: 582

\- Hardhat: 574

\- NO-Mask: 491

\- Safety Vest: 424

\- NO-Hardhat: 402



Some classes contain very few annotations, including:



\- bus: 1

\- fire hydrant: 6

\- mini-van: 7

\- semi: 7

\- truck: 7

\- SUV: 16



This indicates significant class imbalance within the complete dataset.



The generated class distribution is available at:



`results/plots/class\_distribution.png`



\## PPE-Relevant Classes



Although the dataset contains 25 classes, the main PPE compliance problem focuses particularly on classes such as:



\- Person

\- Hardhat

\- NO-Hardhat

\- Safety Vest

\- NO-Safety Vest

\- Mask

\- NO-Mask

\- Gloves



\## Augmentation Analysis



To increase variation in the training data, the following augmentation techniques were investigated:



1\. Horizontal flipping

2\. Brightness adjustment

3\. Small-angle rotation



These transformations simulate variations that may occur in construction-site imagery, including different worker orientations, lighting conditions, and camera viewpoints.



Example augmentation outputs are available at:



`results/augmentation/augmentation\_comparison.png`



\## Observations



\- The dataset contains 717 images and 5570 object annotations.

\- All images have corresponding annotation files.

\- The dataset is strongly imbalanced across its 25 classes.

\- Person is the most represented class.

\- Several non-PPE classes contain very few examples.

\- Augmentation can increase visual variation during model training.

\- Model performance should be examined per class because overall metrics may hide poor performance on underrepresented classes.

