import numpy as np
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import cv2
import imagehash
import numpy as np
from deepface import DeepFace
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from people_tracking.face_encoding import (
    assign_faces_per_frame,
    get_default_classifier,
    train_classifier,
)

FONT_SCALE = 1e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 3e-4  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 3e-3  # Adjust for larger Y-offset of text and bounding box

placeholder_hash = imagehash.average_hash(
    Image.open("/mnt/castle/processed/placeholder.png")
)  # Load the placeholder image and compute its hash
KNOWN_FACE_DATABASE = "groundtruth"


def is_similar_placeholder(frame_rgb, threshold=1):
    frame_hash = imagehash.average_hash(
        Image.fromarray(frame_rgb)
    )  # Compute hash of the current frame
    # Compare with the placeholder hash
    return (
        frame_hash - placeholder_hash < threshold
    )  # Return True if similar to placeholder


def draw_face_id(frame, face_location, face_id, color=(0, 255, 0)):
    height, width = frame.shape[:2]
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # get position for text based on the face location
    x, y = left, top - 10
    # Ensure the text does not go out of bounds
    y = max(y, 0)  # Ensure y is not negative
    x = max(x, 0)  # Ensure x is not negative

    cv2.putText(
        frame,
        f"{face_id}",
        (x, y - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=color,
    )
    return frame


def save_face_image(output_face, face_image, face_id, frame_path, index=0):
    if not os.path.exists(f"{output_face}/{face_id}"):
        os.makedirs(f"{output_face}/{face_id}")

    person = os.path.basename(os.path.dirname(frame_path))
    timestamp = os.path.basename(frame_path)
    face_image_path = f"{output_face}/{face_id}/{person}_{index}_{timestamp}"
    # if blur_detection(face_image):
    #     print(f"Skipping blurred face image for ID {face_id} in frame {frame_path}")
    #     return
    cv2.imwrite(face_image_path, face_image)


def name_to_seconds(name):
    name = name.split(".")[0]  # Remove file extension
    hour, seconds = name.split("_")
    hour = int(hour)
    seconds = int(seconds)
    return hour * 3600 + seconds


def box_center(box):
    top, right, bottom, left = box
    return ((left + right) // 2, (top + bottom) // 2)


def center_dist(b1, b2):
    c1 = box_center(b1)
    c2 = box_center(b2)
    return np.linalg.norm(np.array(c1) - np.array(c2))




def post_process(raw_detections, window=5, dist_thresh=100):
    # Flatten the detections list
    detections = [d for frame in raw_detections for d in frame["faces"]]

    # dict of frame_path, index, new_id
    changes = defaultdict(dict)

    for i in range(len(detections)):
        current = detections[i]
        candidates = []
        for j in range(max(0, i - window), min(len(detections), i + window + 1)):
            if i == j:
                continue
            if (
                detections[j]["frame"] != current["frame"]
            ):  # Optional: compare same timestamp if frame is duplicated
                continue
            if center_dist(current["box"], detections[j]["box"]) < dist_thresh:
                candidates.append(detections[j]["id"])

        all_ids = candidates + [current["id"]]
        # remove "Unknown" or None IDs
        all_ids = [id_ for id_ in all_ids if id_ not in (None, "Unknown")]
        if not all_ids:
            all_ids = ["Unknown"]
        # Get the most common ID
        most_common_id = Counter(all_ids).most_common(1)[0][0]
        changes[current["frame"]][current["index"]] = {
            "new_id": most_common_id,
            "raw_id": current["id"],
        }

    # Now create the smoothed detections
    for frame in raw_detections:
        for face in frame["faces"]:
            if face["index"] in changes[face["frame"]]:
                new_id_info = changes[face["frame"]][face["index"]]
                print(
                    f"Face {face['index']} in frame {face['frame']} smoothed from {face['id']} to {new_id_info['new_id']}"
                )
                face.update(
                    {"id": new_id_info["new_id"], "raw_id": new_id_info["raw_id"]}
                )
            else:
                face["raw_id"] = face["id"]
    return raw_detections  # Return the modified detections with smoothed IDs


def main(key, frame_paths):
    NEW_IDENTIFICATION = False  # Set to True to group new faces
    ALLOW_NEW_FACES = False  # or mark as unknown

    clf = None
    all_encodings, names = [], []
    encodings = np.zeros((0, 512))  # Initialize empty encodings array
    if not NEW_IDENTIFICATION:
        clf = get_default_classifier(
            train_dir=KNOWN_FACE_DATABASE, save_file="groundtruth_encodings.pkl"
        )

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print("Error: Could not read the first image.")
        return

    os.makedirs(f"video_reid", exist_ok=True)
    output_video_path = f"video_reid/{key}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    output_face = f"output_faces/{key}"
    if not os.path.exists(output_face):
        os.makedirs(output_face)

    print("Processing images...")
    detections = []
    new_face_count = 0
    unknown_face_count = 0
    skipped_frames = 0

    main_loop = tqdm(total=len(frame_paths), desc="Processing frames", unit="frame")
    try:
        for i, frame_path in enumerate(frame_paths):
            main_loop.update(1)
            person = os.path.basename(os.path.dirname(frame_path))
            if i % 5 != 0:
                continue

            rgb_frame = cv2.imread(frame_path)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

            if is_similar_placeholder(rgb_frame):
                skipped_frames += 1
                main_loop.set_description(f"Skipping placeholder frame {i + 1}")
                continue

            detections.append(
                {
                    "frame": frame_path,
                    "faces": [],
                }
            )

            dfs = DeepFace.represent(
                img_path=frame_path,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="yolov8",
                normalization="Facenet2018",
            )

            predict_probs = []
            frame_face_encodings = []
            for df in dfs:
                confidence = df["face_confidence"]
                threshold = 0.5
                if confidence < threshold:
                    continue
                face = df["facial_area"]
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                top, right, bottom, left = y, x + w, y + h, x
                # if NEW_IDENTIFICATION:
                # remove small faces
                # if w < 100 or h < 100:
                #     continue
                face_encoding = df["embedding"]

                if NEW_IDENTIFICATION:
                    frame_face_encodings.append(face_encoding)
                else:
                    assert clf is not None, "Classifier is not loaded."
                    probs = clf.predict_proba([face_encoding])[0]
                    predict_probs.append(probs)

                # Append detection information
                detections[-1]["faces"].append(
                    {
                        "frame": frame_path,
                        "index": len(detections[-1]["faces"]),
                        "box": (top, right, bottom, left),
                        "id": None,
                    }
                )

            # Process predictions in a way that
            # a person can only be detected once per frame
            if NEW_IDENTIFICATION:
                all_encodings.extend(frame_face_encodings)
            else:
                if not predict_probs:
                    continue
                assert clf is not None, "Classifier is not loaded."
                face_assignments = assign_faces_per_frame(
                    predict_probs, clf.classes_, threshold=0.5
                )
                for face_idx, face_id in face_assignments.items():
                    if face_id is None:
                        if ALLOW_NEW_FACES:
                            face_id = f"{new_face_count:04d}"
                            new_face_count += 1
                            face_encoding = frame_face_encodings[face_idx]
                            encodings = np.vstack([encodings, face_encoding])
                            names.append(face_id)
                            # Retrain the classifier with the new face
                            clf = train_classifier(encodings, names)
                        else:
                            face_id = "Unknown"
                            unknown_face_count += 1
                    detections[-1]["faces"][face_idx]["id"] = face_id

            if NEW_IDENTIFICATION:
                main_loop.set_description(
                    f"{person}: Got {len(all_encodings)} faces, {skipped_frames} skipped"
                )
            elif ALLOW_NEW_FACES:
                main_loop.set_description(
                    f"{person}: {new_face_count} new faces detected"
                )
            else:
                main_loop.set_description(
                    f"{person}: {unknown_face_count} unknown faces detected"
                )

    except KeyboardInterrupt:
        pass

    if NEW_IDENTIFICATION:
        new_detections = []
        print("Training new classifier with detected faces...")
        encodings = np.array(all_encodings).reshape(
            -1, 512
        )  # Reshape to 2D array for clustering
        print(encodings.shape)
        # cluster with KNN without labels
        knn = KMeans(n_clusters=min(len(all_encodings), 13), random_state=42)
        knn.fit(encodings)
        i = 0
        labels = knn.labels_
        assert labels is not None, "Labels are None after clustering."
        print("Clustering complete. Assigning IDs to faces...")
        labels = labels.tolist()
        for detection in detections:
            new_faces = []
            for face in detection["faces"]:
                face_id = f"{labels[i]:04d}"  # Use cluster label as ID
                new_faces.append(
                    {
                        "frame": face["frame"],
                        "box": face["box"],
                        "id": face_id,
                    }
                )
                i += 1
            new_detections.append({"frame": detection["frame"], "faces": new_faces})
        detections = new_detections
        print(f"Total faces detected: {len(encodings)}")

    # Post-process the detections to smooth IDs across frames
    if not NEW_IDENTIFICATION:
        print("Post-processing...")
        detections = post_process(detections, window=5, dist_thresh=50)

    SAVE_VIDEO = True
    print("Saving results...")
    out = cv2.VideoWriter(
        output_video_path, fourcc, 10, (first_frame.shape[1], first_frame.shape[0])
    )
    try:
        for detection in detections:
            frame_path = detection["frame"]
            rgb_frame = cv2.imread(frame_path)
            annotated_frame = rgb_frame.copy()
            for i, face in enumerate(detection["faces"]):
                face_id = face["id"]
                raw_id = face.get("raw_id", face_id)
                box = face["box"]
                top, right, bottom, left = box
                save_face_image(
                    output_face,
                    rgb_frame[top:bottom, left:right],
                    face_id,
                    frame_path,
                    index=i,
                )
                if SAVE_VIDEO:
                    annotated_frame = draw_face_id(
                        annotated_frame, (top, right, bottom, left), face_id
                    )
                    if face_id != raw_id:
                        print(f"Face ID {face_id} smoothed from raw ID {raw_id}")
                        annotated_frame = draw_face_id(
                            annotated_frame,
                            (top, right, bottom, left),
                            f"{face_id} ({raw_id})",
                            color=(0, 0, 255),
                        )

            if SAVE_VIDEO:
                out.write(annotated_frame)
    except KeyboardInterrupt:
        print("Processing interrupted by user.")

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(
        description="Face Recognition and Re-identification"
    )
    parser.add_argument(
        "--path", type=str, help="Path to the folder containing images of faces"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Indicate if multiple folders should be processed",
    )

    args = parser.parse_args()
    if args.path is None:
        print("Please provide the path to the folder containing images of faces.")
        sys.exit(1)

    if not os.path.exists(args.path):
        print(f"Path '{args.path}' does not exist.")
        sys.exit(1)

    frame_paths = []
    if args.multiple:
        key = "All2"
        for person in sorted(os.listdir(args.path)):
            if not os.path.isdir(os.path.join(args.path, person)):
                print(f"Skipping non-directory item: {person}")
                continue
            # main(sys.argv[1] + "/" + person)
            for filename in sorted(
                os.listdir(os.path.join(args.path, person)), key=name_to_seconds
            ):
                if filename.endswith(".png"):
                    img_path = os.path.join(args.path, person, filename)
                    frame_paths.append(img_path)
    else:
        print(f"Processing single folder: {args.path}")
        # get frame from a folder
        key = os.path.basename(args.path)
        for filename in sorted(os.listdir(args.path), key=name_to_seconds):
            if filename.endswith(".png"):
                img_path = os.path.join(args.path, filename)
                frame_paths.append(img_path)
    # get size of the first image
    if not frame_paths:
        print("No images found in the specified folder.")

    main(key, frame_paths)
