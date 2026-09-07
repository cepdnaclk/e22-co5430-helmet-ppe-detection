from pathlib import Path
import matplotlib.pyplot as plt

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Excavator",
    "Gloves",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "SUV",
    "Safety Cone",
    "Safety Vest",
    "bus",
    "dump truck",
    "fire hydrant",
    "machinery",
    "mini-van",
    "sedan",
    "semi",
    "trailer",
    "truck and trailer",
    "truck",
    "van",
    "vehicle",
    "wheel loader"
]

counts = [0] * len(CLASS_NAMES)

for split in ["train", "valid", "test"]:
    labels_dir = DATASET_DIR / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 5:
                    class_id = int(parts[0])

                    if 0 <= class_id < len(CLASS_NAMES):
                        counts[class_id] += 1

print("\n========== CLASS DISTRIBUTION ==========")

for name, count in zip(CLASS_NAMES, counts):
    print(f"{name}: {count}")

print(f"\nTotal annotations: {sum(counts)}")

plt.figure(figsize=(14, 7))
plt.bar(CLASS_NAMES, counts)

plt.xlabel("Class")
plt.ylabel("Number of Annotations")
plt.title("Dataset Class Distribution")

plt.xticks(rotation=75)
plt.tight_layout()

output_file = OUTPUT_DIR / "class_distribution.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"\nChart saved to: {output_file}")