import json
import math
import os
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import logging
from people_tracking.utils import to_sort_key

# Suppress warnings
logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)

model_name = "facebook/dinov2-with-registers-small"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


ignore = []
if os.path.exists("duplicated.txt"):
    with open("duplicated.txt", "r") as f:
        for line in f.readlines():
            ignore.append(line.strip())

ignore = set(ignore)

def compute_features(photos_batch):
    # Load all the photos from the batch
    images = []
    new_photos_batch = []
    for photo_file in photos_batch:
        try:
            photo = Image.open(photo_file)
            images.append(photo)
            new_photos_batch.append(photo_file)
        except:
            print("Corrupted image", photo_file)

    # Process the photos
    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        # Get the embeddings for the first token (CLS token)
        embedding = last_hidden_states[:, 0, :].detach().numpy()
    return new_photos_batch, embedding


if __name__ == "__main__":
    # print("Available models", open_clip.list_pretrained())
    photofiles = []
    photo_keys = []

    batch_size = 64
    photos_path = "/mnt/castle/castle_downloader/keyframes/"
    output_path = f"/mnt/castle/processed/features/{model_name}/"

    def to_key(path):
        return path.replace(photos_path, "")

    print("Looking up photos")
    # all_images = glob(f"{photos_path}/**/samples_5fps/*.jpg", recursive=True)
    all_images = glob(f"{photos_path}/**/*.jpg", recursive=True)
    sort_keys = [to_sort_key(path) for path in all_images]
    # Sort the images by their sort_keys
    sorted_images = sorted(zip(sort_keys, all_images), key=lambda x: x[0])
    all_images = [path for _, path in sorted_images]
    for path in tqdm(all_images):
        key = to_key(path)
        if key in ignore:
            continue
        photofiles.append(path)
        photo_keys.append(key)
    print("Found", len(photofiles), "photos")

    features_path = Path(output_path)
    os.system(f"mkdir -p {output_path}")
    batches = math.ceil(len(photofiles) / batch_size)
    for i in tqdm(range(batches)):
        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        existed = batch_features_path.exists()
        if existed:
            try:
                # Check if the batch is corrupted
                np.load(batch_features_path)
                pd.read_csv(batch_ids_path)
                continue
            except:
                print("Corrupted batch", i)
                existed = False
        try:
            # Select the photos for the current batch
            batch_files = photofiles[i * batch_size : (i + 1) * batch_size]

            # Compute the features and save to a numpy file
            batch_files, batch_features = compute_features(batch_files)
            np.save(batch_features_path, batch_features)

            # Save the photo IDs to a CSV file
            batch_keys = [to_key(photo) for photo in batch_files]
            photo_keys_data = pd.DataFrame(batch_keys, columns=["photo_id"])
            photo_keys_data.to_csv(batch_ids_path, index=False)
        except Exception as e:
            # Catch problems with the processing to make the process more robust
            print(f"Problem with batch {i}")
            raise (e)

    features_list = []
    for path in sorted(features_path.glob("*.npy")):
        try:
            feat = np.load(path)
            features_list.append(feat)
        except Exception as r:
            print("Corrupted file", path)
            raise r

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)

    photo_ids = pd.concat(
        [pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))]
    )
    # get rid of dir path
    photo_ids["photo_id"] = photo_ids["photo_id"].apply(
        lambda x: x.replace(photos_path, "")
    )
    photo_ids.to_csv(features_path / "photo_ids.csv", index=False)
    # os.system(f"rm -rf {output_path}/0*")
