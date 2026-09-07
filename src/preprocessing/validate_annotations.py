from pathlib import Path

# Dataset paths
DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]

# Dataset has 25 classes: IDs 0 to 24
NUM_CLASSES = 25

total_files = 0
total_annotations = 0
errors = []

for split in SPLITS:
    labels_dir = DATASET_DIR / split / "labels"

    if not labels_dir.exists():
        errors.append(f"Missing labels directory: {labels_dir}")
        continue

    for label_file in labels_dir.glob("*.txt"):
        total_files += 1

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_number, line in enumerate(lines, start=1):
                line = line.strip()

                # Ignore empty lines
                if not line:
                    continue

                parts = line.split()

                # YOLO format must contain:
                # class_id x_center y_center width height
                if len(parts) != 5:
                    errors.append(
                        f"{label_file}: line {line_number} "
                        f"has {len(parts)} values instead of 5"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    errors.append(
                        f"{label_file}: line {line_number} "
                        f"contains non-numeric values"
                    )
                    continue

                # Check class ID
                if not 0 <= class_id < NUM_CLASSES:
                    errors.append(
                        f"{label_file}: line {line_number} "
                        f"has invalid class ID {class_id}"
                    )

                # Check normalized coordinates
                values = {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }

                for name, value in values.items():
                    if not 0 <= value <= 1:
                        errors.append(
                            f"{label_file}: line {line_number} "
                            f"has {name}={value}, outside [0, 1]"
                        )

                total_annotations += 1

        except Exception as e:
            errors.append(f"{label_file}: {e}")

print("\n========== YOLO ANNOTATION VALIDATION ==========")
print(f"Label files checked: {total_files}")
print(f"Annotations checked: {total_annotations}")
print(f"Errors found: {len(errors)}")

if errors:
    print("\n---------- ERRORS ----------")
    for error in errors[:50]:
        print(error)

    if len(errors) > 50:
        print(f"\n... and {len(errors) - 50} more errors.")

    print("\n❌ Annotation validation FAILED.")
else:
    print("\n✅ All YOLO annotations are valid.")
    print("✅ Class IDs are within the expected range.")
    print("✅ Bounding-box values are normalized between 0 and 1.")