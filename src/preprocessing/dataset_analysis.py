from pathlib import Path
from collections import Counter
import yaml
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return names


def count_images(folder):
    return len(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def analyze_split(split_name, class_names):
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"

    image_files = [
        file
        for file in images_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_files = list(labels_dir.glob("*.txt"))

    class_counts = Counter()
    total_annotations = 0

    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if 0 <= class_id < len(class_names):
                    class_counts[class_id] += 1
                    total_annotations += 1

    image_stems = {file.stem for file in image_files}
    label_stems = {file.stem for file in label_files}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "annotations": total_annotations,
        "class_counts": class_counts,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
    }


def save_class_distribution(total_class_counts, class_names):
    counts = [total_class_counts.get(i, 0) for i in range(len(class_names))]

    plt.figure(figsize=(14, 8))
    plt.bar(class_names, counts)

    plt.title("Class Distribution - Construction Site Safety Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Annotations")

    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    output_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nClass distribution graph saved to:")
    print(output_path)


def main():
    print("=" * 60)
    print("CONSTRUCTION SITE SAFETY DATASET ANALYSIS")
    print("=" * 60)

    class_names = load_class_names()

    print(f"\nNumber of classes: {len(class_names)}")

    total_class_counts = Counter()
    total_images = 0
    total_annotations = 0

    for split in ["train", "valid", "test"]:
        results = analyze_split(split, class_names)

        total_images += results["images"]
        total_annotations += results["annotations"]
        total_class_counts.update(results["class_counts"])

        print(f"\n--- {split.upper()} ---")
        print(f"Images: {results['images']}")
        print(f"Label files: {results['labels']}")
        print(f"Annotations: {results['annotations']}")
        print(f"Images without labels: {results['missing_labels']}")
        print(f"Labels without images: {results['missing_images']}")

    print("\n" + "=" * 60)
    print("OVERALL DATASET SUMMARY")
    print("=" * 60)

    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")

    print("\nClass Distribution:")

    for class_id, class_name in enumerate(class_names):
        count = total_class_counts.get(class_id, 0)
        print(f"{class_id:2d} | {class_name:<20} | {count}")

    save_class_distribution(total_class_counts, class_names)


if __name__ == "__main__":
    main()