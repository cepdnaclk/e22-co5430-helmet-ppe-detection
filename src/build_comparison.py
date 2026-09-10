import pandas as pd

def load(path, tag):
    df = pd.read_csv(path, comment="#")
    df = df.rename(columns={
        "precision": f"precision_{tag}",
        "recall": f"recall_{tag}",
        "f1": f"f1_{tag}",
        "mAP50": f"mAP50_{tag}",
        "mAP50-95": f"mAP50-95_{tag}",
    })
    return df

baseline = load("C:/ppe/results/metrics/baseline_metrics.csv", "baseline")
no_aug = load("C:/ppe/results/metrics/no_augmentation_metrics.csv", "no_aug")
improved = load("C:/ppe/results/metrics/improved_metrics.csv", "improved")

merged = baseline.merge(no_aug, on="class").merge(improved, on="class")

cols = ["class"]
for tag in ["baseline", "no_aug", "improved"]:
    cols += [f"precision_{tag}", f"recall_{tag}", f"f1_{tag}", f"mAP50_{tag}", f"mAP50-95_{tag}"]
merged = merged[cols]

# Deltas: improved vs baseline (headline comparison)
merged["mAP50_delta_vs_baseline"] = (merged["mAP50_improved"] - merged["mAP50_baseline"]).round(4)
merged["mAP50-95_delta_vs_baseline"] = (merged["mAP50-95_improved"] - merged["mAP50-95_baseline"]).round(4)
merged["mAP50_delta_vs_no_aug"] = (merged["mAP50_improved"] - merged["mAP50_no_aug"]).round(4)

merged.to_csv("C:/ppe/results/metrics/model_comparison.csv", index=False)
print("Saved: C:/ppe/results/metrics/model_comparison.csv")
print(merged.to_string(index=False))