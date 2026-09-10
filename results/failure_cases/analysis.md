# Failure Case Analysis — PPE Detection

This document analyzes four representative failure cases across three model versions: the baseline YOLOv8n model, an initial "improved" model (v1: 20 epochs, aggressive augmentation), and a refined improved model (v2: 50 epochs, tuned augmentation strength). Each case illustrates a distinct type of detection error, and each was re-evaluated after every training run to track whether the failure was resolved, unchanged, or worsened over successive iterations.

**Quantitative summary** (full detail in `results/metrics/baseline_metrics.csv` and `improved_metrics.csv`):

| Metric | Baseline | Improved v1 | Improved v2 |
|---|---|---|---|
| Precision | 0.504 | 0.670 | 0.705 |
| Recall | 0.462 | 0.343 | 0.468 |
| mAP50 | 0.446 | 0.414 | 0.540 |
| mAP50-95 | 0.304 | 0.238 | 0.350 |

v1 (aggressive augmentation, 20 epochs) underperformed baseline on mAP and recall, illustrating that overly strong augmentation combined with a short training schedule can hurt convergence. v2 (tuned, gentler augmentation, 50 epochs) improved on every metric relative to both baseline and v1, and represents the final selected model. The four cases below examine what this progression looked like in practice at the level of individual predictions.

---

## Case 1: Mannequin False Positive with Contradictory Labels
**Files:** `failure_03_mannequin_fp_baseline.jpg` / `_improved.jpg` (v1) / `_improved_v2.jpg`

**Baseline:** A static mannequin was detected as "Person" (0.94), with simultaneous, mutually exclusive labels on the same region — "Safety Vest" (0.33) and "NO-Safety Vest" (0.57) both present, alongside "NO-Mask" (0.69).

**v1:** Unchanged. Same contradiction persisted.

**v2:** The internal contradiction is resolved — only "NO-Safety Vest" (0.75) remains, no longer paired with a conflicting "Safety Vest" label. A "NO-Hardhat" label also appeared, which is at least internally consistent with the rest of the scene. However, the mannequin is still classified as "Person," so the underlying human-vs-non-human confusion remains.

**Interpretation:** This is a genuine, if partial, improvement. Longer training and gentler augmentation seem to have improved the model's confidence calibration enough to suppress one of two conflicting labels rather than emitting both. It has not, however, taught the model to distinguish a human from a human-shaped mannequin — that would likely require adding negative examples of mannequins/statues to the training data, which is a data-side fix rather than a training-configuration one.

---

## Case 2: Contradictory Same-Region Mask Labels
**Files:** `failure_04_mask_contradiction_baseline.jpg` / `_improved.jpg` (v1) / `_improved_v2.jpg`

**Baseline:** "Mask" (0.70) mislocalized on the hardhat, alongside a correctly-placed but lower-confidence "NO-Mask" (0.27) near the actual face.

**v1:** The correct "NO-Mask" detection was dropped entirely; the incorrect "Mask" on the hardhat remained.

**v2:** The mislocalized "Mask" label on the hardhat is still present. A second, overlapping label is also visible in the same region but is not clearly legible in the rendered output. No "NO-Mask" detection near the actual face is visible in this result.

**Interpretation:** Across all three versions, this case has not been resolved — the model consistently misassociates the "Mask" class with the hardhat region rather than the face. This persists despite the overall metric improvements in v2, suggesting the issue is a spatial/localization bias specific to this class rather than something a general augmentation and epoch-budget increase corrects. Class-specific data review (checking whether "Mask" annotations in training data are frequently placed near head coverings) would be a reasonable next diagnostic step.

---

## Case 3: Missed Detection Under Occlusion and Clutter
**Files:** `failure_01_occlusion_baseline.jpg` / `_improved.jpg` (v1) / `_improved_v2.jpg`

**Baseline:** The occluded worker was not detected; a spurious "Person" (0.38) fired on unrelated machinery, alongside a "dump truck" (0.40) false positive.

**v1:** The spurious "Person" and "dump truck" detections were removed. The real worker was still not detected.

**v2:** No detections at all are produced for this image. The real worker remains undetected, but the false positives seen in baseline do not reappear.

**Interpretation:** Across all three versions, the actual target (a heavily occluded worker in a cluttered, low-light industrial scene) has never been detected. The positive trend is a steady reduction in false positives on this image as training improved (baseline: 2 false detections → v1: 0 false detections, still missing target → v2: 0 detections, still missing target). This suggests the model has become more conservative and reliable in what it does report, but occlusion robustness itself remains an unresolved limitation, likely requiring occlusion-specific augmentation (e.g., random erasing/cutout) or additional training examples of similarly cluttered scenes.

---

## Case 4: Missed Helmet Detection Despite Clear Visibility
**Files:** `failure_02_missed_helmet_baseline.jpg` / `_improved.jpg` (v1) / `_improved_v2.jpg`

**Baseline:** Hardhat not detected; "NO-Safety Vest" correctly detected on the same person.

**v1:** Hardhat still not detected; the previously correct "NO-Safety Vest" detection was also lost.

**v2:** Hardhat still not detected. "NO-Safety Vest" reappears, but now as two overlapping, duplicate detections (0.27 and 0.30) instead of one clean detection. A new, unrelated false positive — "truck and trailer" (0.41) — also appears in the background, which was not present in either prior version.

**Interpretation:** This case shows the clearest example of a failure that persists across all three model versions without resolution: the hardhat, despite being clearly visible and unoccluded, is never detected by any version of the model. This points toward either a genuine gap in training data for this specific hardhat appearance/angle, or a systematic weakness in the Hardhat class that additional epochs and augmentation tuning have not addressed. The reappearance of "NO-Safety Vest" in v2 (compared to its loss in v1) is a positive sign of the recall recovery reflected in the metrics, though the duplicate-box behavior and the new spurious "truck and trailer" detection indicate v2 has not fully eliminated noise, just shifted its form.

---

## Summary of Patterns Across Model Versions

1. **False positives decreased steadily across training iterations.** Case 1 (contradiction resolved) and Case 3 (spurious detections eliminated) both show a consistent trend toward more conservative, reliable outputs as training matured.
2. **Certain class-specific localization errors persisted unchanged across all three models.** Case 2 (Mask mislocalized on hardhat) and Case 4 (Hardhat never detected) suggest these are not epoch-budget or augmentation-strength problems, but more likely reflect gaps or biases in the training data itself for these specific classes/appearances.
3. **The aggregate metric improvement in v2 is real but not uniform at the individual-image level.** Overall mAP50 rose from 0.446 (baseline) to 0.540 (v2), and this is corroborated by genuine improvements in Cases 1 and 3. However, Cases 2 and 4 show that specific, recurring failure modes were not fixed by longer training and tuned augmentation alone — they appear to require targeted data-level intervention (additional or corrected annotations for the affected classes) rather than further hyperparameter adjustment.
4. **Selected final model:** Improved v2 (50 epochs, tuned augmentation: `hsv_s=0.4`, `degrees=5.0`, `translate=0.1`, `scale=0.3`) is adopted as the project's improved model, based on its consistent superiority across all four aggregate metrics and its demonstrated reduction in false-positive noise in the qualitative failure analysis above.
