# Deduplicate the images by features / time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
from pathlib import Path

from imagededup.methods import PHash

from dotenv import load_dotenv
from tqdm import tqdm
from utils.placeholder import is_Image_similar_placeholder

load_dotenv()
KEYFRAMES_DIR = os.getenv("KEYFRAMES_DIR", "keyframes")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed")

# --- 1. Remove placeholder images
def process_image(img_path: Path):
    try:
        image = Image.open(str(img_path)).convert("L")
        if is_Image_similar_placeholder(image):
            os.remove(img_path)
            return 1
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return 0

def remove_placeholder_images(image_dir: Path):
    all_images = list(image_dir.glob("**/*.jpg"))
    print(f"Scanning {len(all_images):,} images in {image_dir}...")

    removed = 0
    for img_path in tqdm(all_images, desc="Removing placeholders"):
        removed += process_image(img_path)

    # NUM_THREADS = 8
    # with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    #     results = list(tqdm(
    #         executor.map(process_image, all_images, chunksize=100),
    #         total=len(all_images),
    #         desc="Removing placeholders"
    #     ))
    #     removed = sum(results)

    print(f"✅ Removed {removed:,} placeholder images.")

# --- 2. Find duplicates and save them to a file
def find_duplicates(image_dir: Path):
    """Find duplicate images in the given directory."""
    phasher = PHash()
    duplicates = phasher.find_duplicates_to_remove(Path(image_dir), recursive=True)
    print(f"Found {len(duplicates)} duplicates")

    # Save
    with open(Path(PROCESSED_DIR) / "duplicates.txt", "w") as f:
        for img in duplicates:
            f.write(img + "\n")

if __name__ == "__main__":
    keyframes_dir = Path(KEYFRAMES_DIR)

    # Remove placeholder images
    remove_placeholder_images(keyframes_dir)

    # Find duplicates
    find_duplicates(keyframes_dir)
