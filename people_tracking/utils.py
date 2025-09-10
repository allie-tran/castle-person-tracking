import csv
import time
import os
import random

import cv2
import numpy as np
import requests
from rich import print
from .llm import LLM, MixedContent, get_visual_content

from utils.colors import PEOPLE_COLORS


def to_sort_key(image):
    image = image.split(".")[0]
    rest, time = image.rsplit("/", 1)
    hour, seconds = time.split("_")
    hour = int(hour)
    seconds = int(seconds)
    return f"{rest}/{hour * 3600 + seconds:06d}"

def retry_with_wait(func, retries=3, wait_time=1, *args, **kwargs):
    """
    Retry a function with a specified number of retries and wait time between attempts.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    raise RuntimeError(f"Function {func.__name__} failed after {retries} attempts.")


# Dictionary to store colors for unique person IDs for consistent visualization
unique_person_colors = {
    person: tuple(int(h.strip("#")[i : i + 2], 16) for i in (0, 2, 4))
    for person, h in PEOPLE_COLORS.items()
}


def get_id_color(unique_id):
    if unique_id not in unique_person_colors:
        # Generate a random color (B, G, R)
        unique_person_colors[unique_id] = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200),
        )
    return unique_person_colors[unique_id]


def save_raw_tracking_output(
    metadata_dir, frame_id, track_id, bbox, person_feat, face_bbox, potential_id, mask
):
    """
    save_dir: directory to save tracking output images.
    Creates the directory if it doesn't exist.
    """
    metadata_path = os.path.join(metadata_dir, "metadata.csv")
    csv_file = open(metadata_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    # Inside your frame loop, after detecting a person and getting their mask and box:
    x1, y1, x2, y2 = bbox  # Bounding box

    # Save mask
    mask_path = None
    if mask is not None:
        mask_filename = f"frame{frame_id:06d}_track{int(track_id):03d}.png"
        mask_path = os.path.join("masks", mask_filename)
        cv2.imwrite(os.path.join(metadata_dir, mask_path), (mask * 255).astype("uint8"))

    # Save feature vector
    feature_filename = f"frame{frame_id:06d}_track{int(track_id):03d}_feat.npy"
    feature_path = os.path.join("features", feature_filename)
    np.save(os.path.join(metadata_dir, feature_path), person_feat)

    if not face_bbox:
        face_bbox = (0, 0, 0, 0)

    face_x1, face_y1, face_x2, face_y2 = face_bbox  # Face bounding box
    # Write metadata
    csv_writer.writerow(
        [
            frame_id,
            int(track_id),
            x1,
            y1,
            x2,
            y2,
            feature_path,
            face_x1,
            face_y1,
            face_x2,
            face_y2,
            potential_id,
            mask_path,
        ]
    )


def load_raw_tracking_output(save_dir):
    """
    save_dir: directory where tracking output images are saved.
    Returns a list of dictionaries with metadata for each tracked person.
    """
    metadata = []
    csv_file_path = os.path.join(save_dir, "metadata.csv")
    if not os.path.exists(csv_file_path):
        return metadata

    with open(csv_file_path, "r", newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            metadata.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "track_id": int(float(row["track_id"])),
                    "bbox": (
                        int(row["x1"]),
                        int(row["y1"]),
                        int(row["x2"]),
                        int(row["y2"]),
                    ),
                    "feature_path": row["feature_path"],
                    "face_bbox": (
                        int(row["face_x1"]),
                        int(row["face_y1"]),
                        int(row["face_x2"]),
                        int(row["face_y2"]),
                    ),
                    "potential_id": row["potential_id"],
                    "mask_path": row["mask_path"],
                }
            )
    return metadata


def create_metadata_file(
    metadata_dir: str,
    overwrite_existing: bool = False,
):
    """
    Create a metadata file for tracking output.
    key: unique identifier for the tracking session.
    OUTPUT_FOLDER_PATH: base directory to save the metadata file.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(os.path.join(metadata_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(metadata_dir, "features"), exist_ok=True)
    if os.path.exists(os.path.join(metadata_dir, "metadata.csv")):
        if not overwrite_existing:
            print("Metadata file already exists")
            return True
        os.remove(os.path.join(metadata_dir, "metadata.csv"))

    csv_file = open(os.path.join(metadata_dir, "metadata.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame_id",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "feature_path",
            "face_x1",
            "face_y1",
            "face_x2",
            "face_y2",
            "potential_id",
            "mask_path",
        ]
    )
    return False


FONT_SCALE = 6e-4  # Adjust for larger font size in all images
THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 1e-3  # Adjust for larger Y-offset of text and bounding box


def putText(frame, text, pos, color):
    """
    Put text on the frame at the specified position.
    """
    font_scale = FONT_SCALE * min(frame.shape[0], frame.shape[1])
    thickness = int(THICKNESS_SCALE * min(frame.shape[0], frame.shape[1]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = pos[0] - text_size[0] // 2
    text_y = pos[1] - int(text_size[1] * TEXT_Y_OFFSET_SCALE)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


MAX_FRAMES = 1000
llm_model = LLM()


def get_captions_from_frames(
    cv_frames: list[np.ndarray], instructions: list[str]
) -> str:
    if len(cv_frames) == 0:
        return ""

    if len(cv_frames) > MAX_FRAMES:
        # sample frames
        indices = sorted(random.sample(range(len(cv_frames)), MAX_FRAMES))
        cv_frames = [cv_frames[i] for i in indices]

    image_bytes = []
    for frame in cv_frames:
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_rgb = cv2.resize(
        #     frame_rgb, (512, 512), interpolation=cv2.INTER_AREA
        # )
        # Get the buffer of the image
        _, buffer = cv2.imencode(".jpg", frame_rgb)
        image_bytes.append(buffer.tobytes())


    description: str = retry_with_wait(
        get_description_from_frames,
        instructions=instructions,
        image_bytes=image_bytes,
        retries=3,
        wait_time=30,
    )

    # Rewrite description based on the transcript
    rewritten_response: str = retry_with_wait(
        get_rewritten_description,
        description=description,
        instructions=instructions,
        retries=3,
        wait_time=30,
    )
    return rewritten_response if rewritten_response else description.strip()

USE_TARSIER = False
def get_description_from_frames(instructions: list[str], image_bytes: list[bytes]) -> str:
    if USE_TARSIER:
        # Get description first
        response = requests.post(
            "http://localhost:8080/describe",
            files=[
                ("images", (f"frame_{i}.jpg", img_bytes, "image/jpeg"))
                for i, img_bytes in enumerate(image_bytes)
            ],
            data={"instruction": instructions[0]},
        )
        if response.status_code != 200:
            print(f"[red]Error getting description:[/red] {response.text}")
        description = response.json().get("description", "")
    else:
        description = llm_model.generate_from_mixed_media(
            get_visual_content(image_bytes) + [MixedContent(type="text", content=instructions[0])],
        )
    return str(description).strip()

def get_rewritten_description(description, instructions: list[str] = []):
    if len(instructions) >= 2 and instructions[1]:
        prompt = f"Rewrite these sentences:\n{description}.\n\n{instructions[1]}"
        rewritten_response: str = llm_model.generate_from_text(prompt)  # type: ignore
        return rewritten_response.strip()
    else:
        # If no instructions are provided, just return the description
        return description

static_cameras = ["Kitchen", "Living1", "Living2", "Meeting", "Reading"]
persons = [
    "Stevan",
    "Bjorn",
    "Allie",
    "Cathal",
    "Luca",
    "Florian",
    "Onanong",
    "Bao",
    "Linh",
    "Tien",
    "Werner",
    "Klaus",
]


def get_prompt_for_camera(camera: str, subtitles: str) -> list[str]:
    instruction: list[str] = []
    if camera in static_cameras:
        instruction.append(
            f"Describe the scene in this room: {camera}. The people are {', '.join(persons)}. Return an empty string if you do not know.",
        )
    else:
        instruction.append(
            f"This is a POV footage from a camera worn by {camera}, being in the same house with {', '.join(persons)}. Describe what they are doing, refer to the people by their names if know. Write the sentences as: <name> is doing <action>."
        )

    if subtitles:
        instruction.append(
            f"Take into account the transcript of the segment. Summarize the topic of the conversation. Here is the transcript of the segment: {subtitles}."
        )
    return instruction
