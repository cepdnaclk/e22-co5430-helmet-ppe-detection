\# Preprocessing and Augmentation Report



\## 1. Dataset Preparation



The Construction Site Safety dataset contains 717 images divided into training,

validation, and testing subsets.



| Split | Images | Labels |

|---|---:|---:|

| Train | 521 | 521 |

| Validation | 114 | 114 |

| Test | 82 | 82 |

| Total | 717 | 717 |



The dataset contains 25 object classes and 5,570 annotated objects.



\## 2. Annotation Validation



The YOLO annotation files were automatically validated before model development.



The validation checked:



\- Correct YOLO annotation format

\- Five values per annotation

\- Valid class IDs from 0 to 24

\- Normalized bounding-box coordinates

\- Values within the range 0 to 1



A total of 717 label files and 5,570 annotations were checked.



\*\*Validation result: 0 errors.\*\*



\## 3. Class Distribution



The dataset contains different numbers of annotations for each class.



The most frequent class is `Person` with 1,148 annotations, while some classes

have very few examples. For example, `bus` has only 1 annotation.



This indicates class imbalance in the dataset and is considered when interpreting

model performance.



The class distribution chart is available at:



`results/plots/class\_distribution.png`



\## 4. Preprocessing



The dataset uses YOLO-format annotations.



Images are resized to 640 × 640 pixels during the YOLO training pipeline.

Bounding-box coordinates remain normalized according to the YOLO annotation format.



The original train, validation, and test sets are kept separate to prevent

data leakage during evaluation.



\## 5. Data Augmentation



Data augmentation is used during model training to improve robustness to

variations commonly found in construction-site environments.



The augmentation strategy considers:



\- Horizontal flipping

\- HSV/color variation

\- Scaling

\- Translation

\- Mosaic augmentation



These transformations can expose the model to changes in lighting, viewpoint,

object position, and scene composition.



Augmentation is applied to the training data only. Validation and test data

remain unchanged for fair model evaluation.



\## 6. Purpose of Augmentation



Construction-site images can contain:



\- Different lighting conditions

\- Different camera viewpoints

\- Occluded workers

\- Different object scales

\- Crowded scenes

\- PPE partially hidden by other objects



The selected augmentation techniques are intended to improve the model's

ability to generalize to these variations.



\## 7. Dataset Quality Summary



The dataset verification found:



\- 717 images

\- 717 corresponding label files

\- 5,570 annotations

\- 25 classes

\- 0 annotation validation errors



Therefore, the dataset is structurally suitable for the subsequent model

training and evaluation stages.

