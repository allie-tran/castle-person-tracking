from tkinter import Image
import cv2
import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from ultralytics import YOLO
import shutil
from people_tracking.people_track_reid import extract_person_features, get_face_data, merge_track_ids
from people_tracking.utils import save_raw_tracking_output, load_raw_tracking_output, create_metadata_file, get_id_color
from people_tracking.face_reid import get_face_embedding_from_crop, classify_face_embedding, PEOPLE_NAMES

# ---------------- CONFIG ----------------
YOLO_MODEL_PATH = "models/yolo11n-seg.pt"
YOLO_FACE_MODEL = "models/yolov12n-face.pt"  # pretrained face model
PERSON_CONF_THRESHOLD = 0.5
FACE_CONF_THRESHOLD = 0.5
FACE_VERIFY_THRESHOLD = 0.5
SMALL_SIZE = (1280, 720)

# ---------------- CLI ----------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--person", required=True)
parser.add_argument("--day", required=True)
parser.add_argument("--hour", required=True)
parser.add_argument("--output_folder", default="tracking_output")
parser.add_argument("--no_video", action="store_true")
args = parser.parse_args()

key = f"{args.day}_{args.person}_{args.hour}"
out_dir = os.path.join(args.output_folder, key)
metadata_dir = os.path.join(out_dir, "intermediate")
os.makedirs(metadata_dir, exist_ok=True)

# ---------------- MODELS ----------------
yolo = YOLO(YOLO_MODEL_PATH, task="segment", verbose=False)
yolo_face = YOLO(YOLO_FACE_MODEL, task="detect", verbose=False)

# ---------------- VIDEO IO ----------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print("Could not open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
create_metadata_file(metadata_dir, overwrite_existing=True)

# ---------------- TRACKING LOOP ----------------
frame_idx = 0
pbar = tqdm(total=total_frames, desc="Tracking")
track_evidence = defaultdict(lambda: {"embeddings": [], "probs": [], "frames": [], "bboxes": []})
skip_seconds = 0
end_seconds = 0

while True:
    try:
        if frame_idx < skip_seconds * fps:
            cap.grab()
            frame_idx += 1
            pbar.update(1)
            continue

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(frame, persist=True, conf=PERSON_CONF_THRESHOLD, classes=0, verbose=False)
        tracking_rows = []
        pred_probs = []

        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None

            for i, track_id in enumerate(ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                features = extract_person_features([crop])
                faces = get_face_data(yolo_face, crop, face_conf_threshold=FACE_CONF_THRESHOLD)
                face_id = None
                probs = [0.0] * len(PEOPLE_NAMES)
                face_bbox = (0, 0, 0, 0)

                if faces:
                    x,y,w,h = faces[0]["bbox"]
                    emb_crop = crop[y:y+h, x:x+w]
                    embedding = get_face_embedding_from_crop(faces[0]["face"])
                    if embedding is None:
                        continue
                    face_id, _ = classify_face_embedding(embedding)
                    face_bbox = faces[0]["bbox"]
                    track_evidence[track_id]["embeddings"].append(embedding)
                    track_evidence[track_id]["probs"].append(probs)

                track_evidence[track_id]["frames"].append(frame_idx)
                track_evidence[track_id]["bboxes"].append((x1, y1, x2, y2))

                tracking_rows.append((metadata_dir, frame_idx, int(track_id), (x1,y1,x2,y2), features, face_bbox, face_id, masks[i] if masks is not None else None))

        for row in tracking_rows:
            save_raw_tracking_output(*row)

        pbar.update(1)
        end_seconds = frame_idx
    except KeyboardInterrupt:
        break

pbar.close()
cap.release()

# ---------------- NORMALIZE IDS ----------------
tracking_data = load_raw_tracking_output(metadata_dir)
mean_embeddings = {tid: np.mean(np.stack(ev["embeddings"]), axis=0)
                   for tid, ev in track_evidence.items() if ev["embeddings"]}
track_id_to_frames = {tid: ev["frames"] for tid, ev in track_evidence.items()}
track_id_to_bboxes = {tid: ev["bboxes"] for tid, ev in track_evidence.items()}
clustered_tracks = merge_track_ids(mean_embeddings, track_id_to_frames, track_id_to_bboxes)

# Assign canonical labels
track_id_to_label = {}
for tid, ev in track_evidence.items():
    embeddings = ev["embeddings"]
    if not embeddings:
        track_id_to_label[tid] = "Unknown"
        continue

    votes = []
    sims = []
    confidences = []

    for e in embeddings:
        pred_label, info = classify_face_embedding(e, sim_threshold=FACE_VERIFY_THRESHOLD, logging=False)
        if pred_label is not None and info["confidence"] >= 0.5:
            votes.append(pred_label)
            sims.append(info['best_sim'])
            confidences.append(info['confidence'])

    if len(votes) < 2:
        track_id_to_label[tid] = "Unknown"
        continue

    label, count = Counter(votes).most_common(1)[0]

    # majority voting
    if count / len(votes) < 0.6:
        track_id_to_label[tid] = "Unknown"
        continue

    track_id_to_label[tid] = label
    
cluster_to_label = {}
for cluster_id, tids in clustered_tracks.items():
    labels = [track_id_to_label[tid] for tid in tids if track_id_to_label[tid] != "Unknown"]
    cluster_to_label[cluster_id] = Counter(labels).most_common(1)[0][0] if labels else "Unknown"

track_id_to_cluster = {tid: cluster_id for cluster_id, tids in clustered_tracks.items() for tid in tids}

# ---------------- FINAL METADATA ----------------
final = defaultdict(list)
for row in tracking_data:
    tid = row["track_id"]
    cluster_id = track_id_to_cluster.get(tid)
    name = cluster_to_label.get(cluster_id, "Unknown")
    row["name"] = name
    final[row["frame_id"]].append(row)

with open(os.path.join(out_dir, "final_metadata.json"), "w") as f:
    json.dump(final, f, indent=2)

# ---------------- FINAL VISUALIZATION ----------------
def draw_label(
    frame,
    text,
    anchor,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.9,
    thickness=2,
    pad=4
):
    """
    anchor: (x, y) reference point (usually top-left of face)
    """
    h_img, w_img = frame.shape[:2]

    (tw, th), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = anchor

    # Prefer: above the anchor
    x_text = x
    y_text = y - th - baseline - pad

    # If above goes out of frame, put below
    if y_text < 0:
        y_text = y + th + baseline + pad

    # If right overflow, shift left
    if x_text + tw > w_img:
        x_text = w_img - tw - pad

    # If left overflow
    if x_text < 0:
        x_text = pad

    # Optional background box (recommended)
    cv2.rectangle(
        frame,
        (x_text - pad, y_text - th - pad),
        (x_text + tw + pad, y_text + baseline + pad),
        color,
        thickness=-1
    )

    cv2.putText(
        frame,
        text,
        (x_text, y_text),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


if not args.no_video:
    cap = cv2.VideoCapture(args.video)
    overlay_video = cv2.VideoWriter(
        os.path.join(out_dir, "processed.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps * 1.5,
        SMALL_SIZE
    )
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Visualizing")
    while True:
        if frame_idx < skip_seconds * fps:
            cap.grab()
            pbar.update(1)
            frame_idx += 1
            continue
        if frame_idx > end_seconds:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        for row in final.get(frame_idx, []):
            x1, y1, x2, y2 = row["bbox"]
            name = row["name"]
            # name = row["potential_id"] # original name before clustering
            if name == "Unknown":
                continue
            color = get_id_color(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            face_bbox = row.get("face_bbox", (0,0,0,0))
            x1f, y1f, w, h = face_bbox
            x1f += x1
            y1f += y1
            x2f = x1f + w
            y2f = y1f + h
            cv2.rectangle(frame, (x1f, y1f), (x2f, y2f), color, 4)
            draw_label(frame, name, (x1f, y1f), color)
        frame = cv2.resize(frame, SMALL_SIZE)

        overlay_video.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    overlay_video.release()
print("Done.")

# Clean up and only keep final metadata and video
shutil.rmtree(metadata_dir)
