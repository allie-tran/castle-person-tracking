from collections import Counter, defaultdict
import json
import cv2
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import os  # Import os for listing directory contents
import sys
from tqdm import tqdm
from rich import print as rprint

from people_tracking.face_encoding import get_default_classifier
from people_tracking.face_reid import assign_faces_per_frame
from people_tracking.gather_subtitles import gather_subtitles, get_subtitles_for_day
from people_tracking.utils import (
    create_metadata_file,
    get_captions_from_frames,
    get_id_color,
    get_prompt_for_camera,
    load_raw_tracking_output,
    putText,
    save_raw_tracking_output,
)
from people_tracking.people_reid import (
    extract_person_features,
    merge_track_ids,
)
from utils.placeholder import is_similar_placeholder
from utils.email import send_email

# --- CONFIGURATION ---
FIXED_SIZE = (1280, 720)  # Fixed size for resizing images
YOLO_MODEL_PATH = "yolo11n-seg.pt"  # or yolov8n-pose.pt
PERSON_CONF_THRESHOLD = 0.5  # Confidence threshold for person detection

# DeepFace specific configurations
FACE_DETECTOR_BACKEND = (
    "yolov8"  # Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
)
FACE_RECOGNITION_MODEL = "Facenet512"  # Options: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'

# IMPORTANT: Update these paths to your trained SVM model and label mapping
KNOWN_FACE_DATABASE = "groundtruth"

# Threshold for face recognition (how similar embeddings must be to be considered the same person)
FACE_VERIFY_THRESHOLD = (
    0.5  # 0.68 = DeepFace's default for ArcFace/cosine. Lower is stricter.
)

# For shot segmentation
SHOT_SEGMENTATION_PATH = "/mnt/castle/processed/webp_segments.json"

# --- Input Configuration ---
# python person_track.py --key <key> <path_to_images> [<output_folder>]
import argparse

parser = argparse.ArgumentParser(description="Person Tracking with YOLO and DeepFace")
parser.add_argument(
    "--person",
    type=str,
    required=True,
    help="Name of the person to track (used for metadata and output naming)",
)
parser.add_argument(
    "--day",
    type=str,
    required=True,
    help="Day of the tracking session (used for metadata and output naming)",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="person_tracking_output",
    help="Path to the output folder for processed images and metadata",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing metadata files",
)
args = parser.parse_args()
OUTPUT_FOLDER_PATH = args.output_folder
day = args.day
person = args.person
key = f"{day}_{person}"
shot_segment_key = f"{day}/{person}"
# image_path = f"/mnt/castle/castle_downloader/keyframes/{day}/{person}"
image_path = f"/mnt/castle/Images/CASTLE/{day}/{person}"
IMAGE_FOLDER_PATH = image_path

# Ensure the image folder exists
if not os.path.exists(IMAGE_FOLDER_PATH):
    print(f"Error: The image folder '{IMAGE_FOLDER_PATH}' does not exist.")
    sys.exit(1)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# --- INITIALIZE MODELS ---
yolo_model = YOLO(
    YOLO_MODEL_PATH, verbose=False, task="segment"
)  # Use 'segment' for segmentation tasks


# --- HELPER FUNCTIONS (UNCHANGED) ---
def get_face_data_from_person_crop(person_crop):
    """
    Detects faces in the person_crop, extracts aligned faces and their embeddings.
    Returns a list of dictionaries: [{'embedding': [], 'bbox': (x1,y1,x2,y2)}]
    """
    face_data = []
    try:
        faces = DeepFace.represent(
            img_path=person_crop,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="yolov8",
            normalization="Facenet2018",
        )

        for face_info in faces:
            confidence = face_info["face_confidence"]
            if confidence < PERSON_CONF_THRESHOLD:
                continue

            face = face_info["facial_area"]
            x, y, w, h = (
                face["x"],
                face["y"],
                face["w"],
                face["h"],
            )
            # Remove box that are the same size (or similar) as the person crop
            size_diff = abs(w - person_crop.shape[1]) + abs(h - person_crop.shape[0])
            if size_diff < 10:  # Adjust threshold as needed
                print(
                    f"Skipping face with size {w}x{h} in person crop of size {person_crop.shape[1]}x{person_crop.shape[0]}"
                )
                continue

            embedding = face_info["embedding"]
            bbox_xyxy = (x, y, x + w, y + h)  # Convert to xyxy format

            face_data.append(
                {
                    "embedding": embedding,
                    "bbox": bbox_xyxy,  # (x1, y1, x2, y2)
                }
            )

    except Exception as e:
        print(f"DeepFace error in get_face_data_from_person_crop: {e}")
    return face_data


# --- Load SVM Classifier and Label Mapping ---
svm_classifier = get_default_classifier(
    train_dir=KNOWN_FACE_DATABASE, save_file="groundtruth_encodings.pkl"
)


def find_or_create_unique_person(new_face_embedding):
    """
    Compares a new face embedding to the known_people_database using DeepFace.verify logic.
    Returns the matching unique_person_id or creates a new one.
    """
    # Use the SVM classifier to predict the person ID based on the new face embedding
    probs = svm_classifier.predict_proba([new_face_embedding])[
        0
    ]  # Get probabilities for each class

    # Get the class with the highest probability
    max_prob_index = np.argmax(probs)
    max_prob_value = probs[max_prob_index]
    class_labels = svm_classifier.classes_
    predicted_person_id = class_labels[max_prob_index]
    if max_prob_value < FACE_VERIFY_THRESHOLD:
        return None, probs
    else:
        # If the predicted person ID is valid, return it
        return predicted_person_id, probs


# Ensure images are sorted correctly to maintain sequence for tracking
image_files = sorted(
    [
        f
        for f in os.listdir(IMAGE_FOLDER_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
)
image_paths = [os.path.join(IMAGE_FOLDER_PATH, f) for f in image_files]
out_video_path = os.path.join(OUTPUT_FOLDER_PATH, key, f"processed.mp4")

os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER_PATH, key), exist_ok=True)
metadata_dir = os.path.join(OUTPUT_FOLDER_PATH, key, "intermediate")
metadata_exists = create_metadata_file(
    metadata_dir=metadata_dir,
    overwrite_existing=args.overwrite,
)

# Load segments

# def webp_name_to_jpg_name(webp_name):
#     """
#     Convert a webp image name to jpg format.
#     """
#     webp_name = webp_name.split(".")[0]
#     day, person, time = webp_name.split("/")
#     hour, seconds = time.split("_")
#     seconds = int(seconds)
#     # 2fps for jpg
#     return [f"{day}/{person}/{hour}_{seconds * 2}.jpg",
#             f"{day}/{person}/{hour}_{seconds * 2 + 1}.jpg"]

shot_segments = []
if os.path.exists(SHOT_SEGMENTATION_PATH):
    with open(SHOT_SEGMENTATION_PATH, "r") as f:
        webp_segments = json.load(f).get(shot_segment_key, [])
        shot_segments = webp_segments
#         # list of "day1/Allie/20_960.webp"
#         shot_segments = []
#         for segment in webp_segments:
#             jpg_segment = []
#             for image in segment:
#                 jpg_segment.extend(webp_name_to_jpg_name(image))
#             shot_segments.append(jpg_segment)
if not shot_segments:
    rprint(
        f"[red]No shot segments found for {shot_segment_key} in {SHOT_SEGMENTATION_PATH}.[/red]"
    )
    exit(1)

print("[blue]Shot segments loaded:[/blue]")
for segment in shot_segments[:10]:
    print(segment)

frame_to_segment = {}
# Create a mapping from frame index to segment ID
for segment_id, segment in enumerate(shot_segments):
    for frame_name in segment:
        # Extract the frame index from the filename
        frame_to_segment[frame_name] = segment_id

# Load subttitles
subtitles = get_subtitles_for_day(day, person)

# --- MAIN IMAGE PROCESSING LOOP ---
track_id_to_names = defaultdict(list)
track_id_to_frames = defaultdict(list)  # Track ID to list of frame indices
to_skips = set()  # Set to track skipped frames
to_skips_path = os.path.join(metadata_dir, "to_skips.txt")
segment_id = 0
print(
    f"[blue]Processing images from {IMAGE_FOLDER_PATH} with {len(image_paths)} images...[/blue]"
)
# --- Load the tracking data ---
tracking_data = load_raw_tracking_output(metadata_dir)
done = set()
rprint(f"[green]Loaded {len(tracking_data)} tracking data rows.[/green]")

for row in tracking_data:
    track_id = row["track_id"]
    potential_id = row["potential_id"]
    track_id_to_names[track_id].append(potential_id)
    track_id_to_frames[track_id].append(row["frame_id"])
    done.add(row["frame_id"])

if os.path.exists(to_skips_path):
    with open(to_skips_path, "r") as f:
        to_skips = {int(line.strip()) for line in f if line.strip().isdigit()}

rprint(
    f"[blue]Loaded {len(to_skips)} skipped frames from {to_skips_path}.[/blue]"
)

def skip_frame(frame_idx):
    to_skips.add(frame_idx)
    with open(to_skips_path, "a") as f:
        f.write(f"{frame_idx}\n")

# --- Processing images and creating track IDs ---
pbar = tqdm(total=len(image_paths), desc="Creating track IDs", unit="image(s)")
try:
    for frame_idx, image_path in enumerate(image_paths):
        pbar.update(1)
        if frame_idx in to_skips or frame_idx in done:
            pbar.set_description(f"Skipping frame {frame_idx} (already processed)")
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            skip_frame(frame_idx)
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        frame = cv2.resize(frame, FIXED_SIZE, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if is_similar_placeholder(frame_rgb):
            skip_frame(frame_idx)
            pbar.set_description(f"Skipping placeholder-like image")
            continue

        pbar.set_description(f"Processing {os.path.basename(image_path)}")

        # Get YOLOv8 detections, tracks, AND MASKS
        # The 'results' object will contain masks if yolov8n-seg.pt is used
        results = yolo_model.track(
            frame, persist=True, conf=PERSON_CONF_THRESHOLD, classes=0, verbose=False,
        )

        # Initialize a blank canvas for drawing all segmentation masks in this frame
        segmented_frame = np.zeros_like(frame, dtype=np.uint8)

        # Store the predicted probabilities for each face
        pred_probs = []
        tracking_output = []

        if results[0].boxes.id is not None:  # type: ignore
            track_ids = results[0].boxes.id.cpu().numpy()  # type: ignore
            xyxys = results[0].boxes.xyxy.cpu().numpy()  # type: ignore
            confs = results[0].boxes.conf.cpu().numpy()  # type: ignore
            clss = results[0].boxes.cls.cpu().numpy()  # type: ignore

            # Access the raw mask data directly from the results object
            # results[0].masks.data is a torch.Tensor (N, H_model, W_model)
            # Convert to numpy once and slice later
            all_masks_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None  # type: ignore

            # Get original frame dimensions for resizing masks
            original_h, original_w = frame.shape[:2]

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, xyxys[i])
                conf = confs[i]
                cls_name = yolo_model.names[int(clss[i])]

                if cls_name == "person":
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if y2 <= y1 or x2 <= x1:
                        continue

                    # apply the mask (if available) to the person crop
                    person_only = frame.copy()
                    if all_masks_data is not None and i < len(all_masks_data):
                        mask_data = all_masks_data[i]
                        # Resize the mask to the original frame size
                        mask_resized = cv2.resize(
                            mask_data,
                            (original_w, original_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        # Expand the mask to make sure the person is fully covered
                        # by dilating the mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

                        mask_dilated = cv2.dilate(mask_resized, kernel, iterations=1)

                        current_mask_binary = (
                            mask_dilated > 0.5
                        )  # Convert to binary mask
                        person_only[~current_mask_binary] = (
                            0  # Set non-mask areas to black
                        )

                    person_crop = person_only[y1:y2, x1:x2]
                    person_feat = extract_person_features([person_crop])
                    face_data_in_crop = get_face_data_from_person_crop(person_crop)
                    potential_id = None
                    face_bboxes = []
                    prob = [0.0 for _ in range(len(svm_classifier.classes_))]

                    for face_info in face_data_in_crop:
                        face_emb = face_info["embedding"]
                        potential_id, prob = find_or_create_unique_person(face_emb)
                        face_bboxes = face_info["bbox"]
                        break  # Only take the first detected face

                    pred_probs.append(prob)
                    tracking_output.append((
                        metadata_dir,
                        frame_idx,
                        track_id,
                        (x1, y1, x2, y2),
                        person_feat,
                        face_bboxes,
                        potential_id,
                        all_masks_data[i] if all_masks_data is not None else None,
                    ))

        # Process the tracking output
        if len(pred_probs) > 0:
            face_assignments = assign_faces_per_frame(
                pred_probs,
                svm_classifier.classes_,
            )

            # Save the tracking output
            for i, data in enumerate(tracking_output):
                face_id = face_assignments.get(i, data[6])
                data = data[:6] + (face_id,) + data[7:]

                save_raw_tracking_output(*data)
                track_id = data[2]

                # Store the detected person information
                track_id_to_names[track_id].append(face_id)
                track_id_to_frames[track_id].append(frame_idx)
        else:
            skip_frame(frame_idx)
            continue

except KeyboardInterrupt:
    print("Processing interrupted by user.")
    rprint("[orange]Continuing to save current progress...[/orange]")

pbar.close()

load_raw_tracking_output(metadata_dir)
# --- Group track IDs based on the embeddings and temporal proximity
rprint(
    "[blue]Grouping track IDs based on person embeddings and temporal proximity...[/blue]"
)
all_features = [row["feature_path"] for row in tracking_data if row["feature_path"]]
if not all_features:
    rprint("[red]No features found in tracking data. Exiting...[/red]")
    sys.exit(1)

# Extract features for all crops
features = [
    np.load(os.path.join(metadata_dir, f))
    for f in all_features
    if os.path.exists(os.path.join(metadata_dir, f))
]
features = np.stack(features) if features else np.empty((0, 0))
features = features.reshape(features.shape[0], -1)  # Ensure 2D shape
print(f"Extracted features shape: {features.shape}")

# get mean features per track to create dict[int, np.ndarray]
track_ids = [row["track_id"] for row in tracking_data]
bboxes = [row["bbox"] for row in tracking_data]
mean_features = defaultdict(list)
for track_id, feature in zip(track_ids, features):
    mean_features[track_id].append(feature)
mean_features = {
    track_id: np.mean(np.array(feats), axis=0) for track_id, feats in mean_features.items()
}
track_id_to_bboxes = defaultdict(list)
for track_id, bbox in zip(track_ids, bboxes):
    track_id_to_bboxes[track_id].append(bbox)

# Cluster the track IDs based on the extracted features
clustered_track_ids = merge_track_ids(
    mean_features,
    track_id_to_frames,
    track_id_to_bboxes,
)

print(
    f"Clustered {len(clustered_track_ids)} track IDs based on features and temporal proximity."
)

# Create name for clusters
cluster_to_names = defaultdict(str)
track_id_to_cluster = {}
for cluster_id, track_ids in clustered_track_ids.items():
    if not track_ids:
        continue
    # Get the names for the first track ID in the cluster
    all_names = []
    for track_id in track_ids:
        if track_id in track_id_to_names:
            all_names.extend(track_id_to_names[track_id])

    all_names = [name for name in all_names if name and name != "Unknown"]
    if all_names:
        # Get the most common name in the cluster
        most_common_name, _ = Counter(all_names).most_common(1)[0]
        cluster_to_names[cluster_id] = most_common_name

    track_id_to_cluster.update(
        {track_id: cluster_id for track_id in track_ids}
    )


# --- CREATE FINAL METADATA ---
# so that we have a mapping of frame_idx to:
# - a list of track_ids
# - each track_id has 1 name, 1 bbox, and 1 face_bbox
final_metadata_path = os.path.join(OUTPUT_FOLDER_PATH, key, "final_metadata.json")
final_metadata = [{
        "image_path": image_path,
        "track_ids": [],
    } for image_path in image_paths
                  ]

for row in tracking_data:
    frame_id = row["frame_id"]
    track_id = row["track_id"]
    name = cluster_to_names.get(
        track_id_to_cluster.get(track_id, None), "Unknown"
    )
    # Ensure the frame_id is within bounds
    if frame_id < len(final_metadata):
        final_metadata[frame_id]["track_ids"].append({
            "name": name,
            **row
        })

# Save the final metadata to a JSON file
import json
with open(final_metadata_path, "w") as f:
    json.dump(final_metadata, f, indent=4)

# --- VISUALIZATION SETUP ---
rprint("[blue]Setting up visualization...[/blue]")
# --- NEW: Iterate through the CSV rows and frames at the same time and visualize ---
fps = 8
overlay_video = cv2.VideoWriter(
    out_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
    fps,  # Frame rate
    FIXED_SIZE,  # Frame size
)
mask_video = cv2.VideoWriter(
    os.path.join(OUTPUT_FOLDER_PATH, key, "masks.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
    fps,  # Frame rate
    FIXED_SIZE,  # Frame size
)

rprint(f"[blue]Output video will be saved to {out_video_path}[/blue]")
current_segment_id = 0
current_frames: list[str] = []  # List to store frames for the current segment
current_overlays: list[np.ndarray] = []
descriptions = []

def to_key(frame_idx) -> str:
    """
    Convert a frame index to a key for the image paths.
    """
    key = image_paths[frame_idx].split("/")[-3:]
    key = "/".join(key)
    return key


try:
    for segment_id, segment in enumerate(shot_segments):
        pbar.set_description(
            f"Processing segment {segment_id + 1}/{len(shot_segments)}"
        )
        current_frames = []
        current_overlays = []  # Reset current overlays for the new segment
        for image_name in segment:
            current_frame_idx = image_paths.index(
                os.path.join(IMAGE_FOLDER_PATH, image_name.split("/")[-1])
            )
            # Reset the frame and segmented_frame for the new segment
            frame = cv2.imread(image_paths[current_frame_idx])
            if frame is None:
                rprint(f"[red]Could not read image {image_paths[current_frame_idx]}.[/red]")
                continue
            frame = cv2.resize(frame, FIXED_SIZE, interpolation=cv2.INTER_LINEAR)
            segmented_frame = np.zeros_like(frame, dtype=np.uint8)
            alpha = 0.5  # Transparency factor for overlay
            for track in final_metadata[current_frame_idx]["track_ids"]:
                track_id = track["track_id"]
                person_bbox = track["bbox"]
                face_bbox = track["face_bbox"]
                display_name = cluster_to_names.get(
                    track_id_to_cluster.get(track_id, None), "Unknown"
                )
                color = get_id_color(display_name)

                # Draw the bounding boxes
                x1, y1, x2, y2 = person_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                fx1, fy1, fx2, fy2 = face_bbox
                if (fx1, fy1, fx2, fy2) != (0, 0, 0, 0):
                    # Draw the face bounding box
                    abs_fx1 = x1 + fx1
                    abs_fy1 = y1 + fy1
                    abs_fx2 = x1 + fx2
                    abs_fy2 = y1 + fy2
                    cv2.rectangle(
                        frame, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), (0, 255, 255), 2
                    )
                # Draw the name just below the bounding box (top-left corner)
                putText(
                    frame,
                    f"{display_name} ({track_id})",
                    (x1, y2),
                    (0, 255, 0),  # Green text
                )
            # Add the current frame to the overlay
            overlay = cv2.addWeighted(frame, 1, segmented_frame, alpha, 0)
            # Write the final frame to the output video
            overlay_video.write(overlay)
            mask_video.write(segmented_frame)
            current_overlays.append(overlay)
            current_frames.append(image_paths[current_frame_idx])

        # Check if we have any frames to process
        if not current_frames:
            rprint(f"[red]No frames found for segment {segment_id}. Skipping...[/red]")
            continue
        # Get the description for the current segment
        pbar.set_description(
            f"Getting description for segment {segment_id + 1} ({len(current_frames)} frames)"
        )
        description = get_captions_from_frames(
            current_overlays,
            get_prompt_for_camera(person, gather_subtitles(current_frames, subtitles)),
        )
        descriptions.append((segment_id, shot_segments[segment_id][0], description))
        rprint(
            f"[green]Segment {segment_id} description: {description}[/green]"
        )


except KeyboardInterrupt:
    print("[orange]Processing interrupted by user.[/orange]")
except Exception as e:
    rprint("[red]An error occurred during processing:[/red]", e)

rprint("[blue]Saving current progress...[/blue]")
overlay_video.release()
mask_video.release()
cv2.destroyAllWindows()

import csv
descriptions_path = os.path.join(OUTPUT_FOLDER_PATH, key, "descriptions.csv")
with open(descriptions_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["segment_id", "segment_name", "description"])
    for segment_id, segment_name, description in descriptions:
        csv_writer.writerow([segment_id, segment_name, description])

send_email(
    subject=f"Person Tracking Completed for {day} - {person}",
    body=f"Tracking completed for {day} - {person}. Metadata saved to {final_metadata_path}.",
)
