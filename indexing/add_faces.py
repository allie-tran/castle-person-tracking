import os
import json
from utils.colors import PEOPLE_COLORS
from pymongo import MongoClient
from tqdm import tqdm

PEOPLE = list(PEOPLE_COLORS.keys())
PEOPLE_COLORS["Unknown"] = "#ffffff"  # Default color for unknown faces
client = MongoClient("localhost", 27017)
db = client["castle"]
collection = db["images"]

image_dir = "/mnt/castle/Images/CASTLE/"

TRACKING_OUTPUT_DIR = "person_tracking_output"
for folder in os.listdir(TRACKING_OUTPUT_DIR):
    person = folder.split("_")[1]
    print(f"Processing folder: {folder} for person: {person}")
    if person in PEOPLE:
        metadata_file = os.path.join(TRACKING_OUTPUT_DIR, folder, "final_metadata.json")
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for frame_data in tqdm(metadata):
                image_path = frame_data["image_path"]
                image_key = image_path.replace(image_dir, "")
                track_ids = frame_data["track_ids"]

                data = []
                for track in track_ids:
                    name = track["name"]
                    bbox = track["face_bbox"] or track["bbox"]
                    data.append({
                        "name": name,
                        "bbox": bbox,
                        "color": PEOPLE_COLORS[name],
                    })

                collection.update_one(
                    {"image": image_key},
                    {"$set": {
                        "people_present": [name["name"] for name in data],
                        "faces": data,
                    }},
                    upsert=True
                )

        except FileNotFoundError:
            print(f"Metadata file not found for {folder}: {metadata_file}")
            continue
