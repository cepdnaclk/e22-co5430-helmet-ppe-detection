from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_IMAGES_DIR = DATASET_DIR / "train" / "images"

OUTPUT_DIR = PROJECT_ROOT / "results" / "augmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_random_image():
    image_files = [
        file
        for file in TRAIN_IMAGES_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise FileNotFoundError("No training images found.")

    return random.choice(image_files)


def horizontal_flip(image):
    return cv2.flip(image, 1)


def brightness_adjustment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = cv2.convertScaleAbs(v, alpha=1.0, beta=40)

    adjusted_hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)


def rotation(image, angle=10):
    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT
    )


def save_comparison(original, flipped, brightened, rotated):
    images = [
        original,
        flipped,
        brightened,
        rotated
    ]

    titles = [
        "Original",
        "Horizontal Flip",
        "Brightness Adjustment",
        "Rotation"
    ]

    plt.figure(figsize=(14, 10))

    for index, image in enumerate(images):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, index + 1)
        plt.imshow(image_rgb)
        plt.title(titles[index])
        plt.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "augmentation_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Augmentation comparison saved to:")
    print(output_path)


def main():
    print("=" * 60)
    print("DATA AUGMENTATION ANALYSIS")
    print("=" * 60)

    image_path = get_random_image()

    print(f"\nSelected image:")
    print(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    flipped = horizontal_flip(image)

    brightened = brightness_adjustment(image)

    rotated = rotation(image)

    save_comparison(
        image,
        flipped,
        brightened,
        rotated
    )

    print("\nAugmentations applied:")
    print("- Horizontal flip")
    print("- Brightness adjustment")
    print("- Rotation")


if __name__ == "__main__":
    main()
