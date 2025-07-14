import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from glob import glob
from typing import Any
from zoneinfo import ZoneInfo

from blurhash import encode
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from PIL import Image
from pymongo import ASCENDING, IndexModel, MongoClient
from pymongo.errors import BulkWriteError
from tqdm.auto import tqdm

load_dotenv()
client = MongoClient("localhost", 27017)
db = client["castle"]
es = Elasticsearch("http://localhost:9200")
image_dir = "/mnt/castle/Images/CASTLE/"
max_workers = 8
batch_size = 1000
ASPECT_RATIO = 16 / 9
THUMBNAIL_HEIGHT = 100

# === UTILITIES ===
path_re = re.compile(r"day(\d+)/[^/]+/(\d+)_(\d+)\.webp$")
pseudodate = datetime(2025, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))


def encode_blurhash(
    image_path: str, aspect_ratio: float, x_components: int = 4, y_components: int = 3
) -> str:
    image = Image.open(image_path)
    image.thumbnail((THUMBNAIL_HEIGHT * aspect_ratio, THUMBNAIL_HEIGHT))
    return encode(image, x_components, y_components)


def image_to_datetime(image_path: str) -> datetime:
    match = path_re.search(image_path)
    if not match:
        raise ValueError(f"Invalid format: {image_path}")
    day, hour, seconds = map(int, match.groups())
    minute, seconds = divmod(seconds, 60)
    if minute >= 60:
        hour += minute // 60
        minute %= 60
        print(f"Adjusted hour to {hour} and minute to {minute} for {image_path}")
    return pseudodate.replace(day=day, hour=hour, minute=minute, second=seconds)


# === MAIN FUNCTIONALITY ===
def index_images():
    collection = db["images"]
    collection.drop()
    collection.create_indexes([IndexModel([("image", ASCENDING)], unique=True)])

    images = sorted(
        glob(f"{image_dir}/**/*.webp", recursive=True), key=os.path.getmtime
    )
    if not images:
        print("No images found.")
        return

    def process(image_path: str) -> dict[str, Any]:
        key = image_path.replace(image_dir, "")
        time = image_to_datetime(image_path)
        return {
            "image": key,
            "aspect_ratio": ASPECT_RATIO,
            "hash_code": encode_blurhash(image_path, ASPECT_RATIO),
            "local_time": time,
            "utc_time": time.astimezone(ZoneInfo("UTC")),
            "day": time.day,
            "person": key.split("/")[1],
        }

    batch = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for data in tqdm(executor.map(process, images), total=len(images)):
            batch.append(data)
            if len(batch) >= batch_size:
                try:
                    collection.insert_many(batch, ordered=False)
                except BulkWriteError as bwe:
                    print("Batch error:", bwe.details)
                batch.clear()
        if batch:
            collection.insert_many(batch, ordered=False)

    print(f"Inserted {len(images)} records.")


# === ENTRY POINT ===
if __name__ == "__main__":
    index_images()
