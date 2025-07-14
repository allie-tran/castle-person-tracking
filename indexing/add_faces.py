import os
from utils.colors import PEOPLE_COLORS

PEOPLE = list(PEOPLE_COLORS.keys())

# For days 2 (jpg)
TRACKING_OUTPUT_DIR = "tracking_output_jpg"
for folder in os.listdir(TRACKING_OUTPUT_DIR):
    if folder.startswith("day2_"):
        person = folder.split("_")[1]
        if person in PEOPLE:
            metadata = os.path.join(TRACKING_OUTPUT_DIR, folder, "intermediate", "metadata.csv")


