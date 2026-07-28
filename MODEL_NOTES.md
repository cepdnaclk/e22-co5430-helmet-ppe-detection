\# Model Notes — Baseline YOLOv8n



\## Configuration

\- Model used: YOLOv8n (pretrained, Ultralytics)

\- Epochs: 20

\- Image size: 640

\- Batch: 16

\- Hardware: CPU (12th Gen Intel Core i7-1260P), no GPU

\- Training time: \~1.5 hours (interrupted once after epoch 7, resumed same day/next day)



\## Results (all 25 classes, validation set)

\- Precision: 0.504

\- Recall: 0.462

\- mAP@0.5: 0.446

\- mAP@0.5:0.95: 0.304



\## Per-class results (PPE-relevant classes)

| Class | Precision | Recall | mAP50 |

|---|---|---|---|

| Person | 0.765 | 0.707 | 0.773 |

| Hardhat | 0.745 | 0.620 | 0.680 |

| NO-Hardhat | 0.557 | 0.551 | 0.562 |

| Safety Vest | 0.614 | 0.634 | 0.625 |

| NO-Safety Vest | 0.644 | 0.580 | 0.619 |

| Mask | 0.684 | 0.714 | 0.785 |

| NO-Mask | 0.527 | 0.391 | 0.340 |



\## Current Problems



\- Small/distant objects missed or given low confidence (e.g. distant trucks, workers in low-light tunnel scene)

\- Mask / NO-Mask class confusion — model repeatedly outputs both labels on the same face, or predicts "Mask" with no mask present

\- Vehicle-related classes (sedan, vehicle, trailer, machinery) generate frequent low-confidence, overlapping, or misclassified boxes in cluttered background scenes — these classes are outside our project's PPE-compliance scope and appear to add noise to detection output

\- Dataset class imbalance — 25 total classes but only \~7 are relevant to our helmet/vest/mask compliance goal; irrelevant classes may be diluting model focus

\- Duplicate/overlapping bounding boxes in multi-person scenes, with inconsistent confidence across near-identical detections

\- Occlusion and poor lighting (e.g. dark tunnel construction scene) reduce detection confidence and cause missed Hardhat detections despite clear visibility to a human viewer



\## Next steps to discuss with team

\- Consider filtering dataset to PPE-relevant classes only (Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask) for the fine-tuned/improved model, based on baseline evidence above

\- Investigate Mask/NO-Mask label quality in source annotations

