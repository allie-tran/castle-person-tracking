from typing import Union
from deepface import DeepFace
from inputimeout import inputimeout, TimeoutOccurred
import numpy as np
import os
import pickle
from scipy.optimize import linear_sum_assignment
from sklearn import svm
from tqdm import tqdm

def get_encodings(train_dir, save_file=None):
    """
    train_dir should contain subdirectories for each person,
    with images of that person inside.

    subdirectories should be named [person_name]_[number],
    e.g. "John_1", "John_2", etc.

    this allows one person to have multiple folder names,
    """
    if save_file is not None and os.path.isfile(save_file):
        try:
            yes = inputimeout(
                f"""Face encodings file {save_file} already exists.
Type 'y' to load from file, or any other key to generate new encodings: """,
                timeout=5
            ).strip().lower()
        except TimeoutOccurred:
            yes = "y" # Default to loading if no input is given

        if yes == "y":
            with open(save_file, "rb") as f:
                names, encodings = pickle.load(f)
            print("Face encodings loaded from file.")
            return encodings, names

    encodings = []
    names = []

    print(f"Generating face encodings from training directory: {train_dir}")
    # Loop through each person in the training directory
    for subdir in sorted(os.listdir(train_dir)):
        if subdir in ["__pycache__", ".DS_Store"]:
            continue

        subdir_path = os.path.join(train_dir, subdir)
        for person in sorted(os.listdir(subdir_path)):
            print(f"Processing folder: {subdir}/{person}")
            if person in ["__pycache__", ".DS_Store"]:
                continue

            pix = os.listdir(os.path.join(train_dir, subdir, person))
            name = person.split("_")[0]  # Get the person's name from the folder name
            # Loop through each training image for the current person
            for person_img in tqdm(pix):
                try:
                    # Get the face encodings for the face in each image file
                    feature = DeepFace.represent(
                        img_path=os.path.join(train_dir, subdir, person, person_img),
                        model_name="Facenet512",
                        enforce_detection=False,
                        detector_backend="skip",
                        normalization="Facenet2018",
                    )
                    if not feature:
                        print(f"No face found in {person}/{person_img}, skipping.")
                        continue
                    encodings.append(feature[0]["embedding"])
                    names.append(name)
                except Exception:
                    continue

    # Save names and encodings to a file
    if save_file is not None:
        with open(save_file, "wb") as f:
            pickle.dump((names, encodings), f)
    return encodings, names

def train_classifier(encodings, names):
    # Create and train the SVC classifier
    # clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf = svm.SVC(probability=True, random_state=42)
    clf.fit(encodings, names)
    return clf

def get_default_classifier(train_dir, save_file=None):
    """
    train_dir: directory containing subdirectories of training images
    save_file: optional file to save/load encodings and classifier

    Returns: classifier, names
    """
    encodings, names = get_encodings(train_dir, save_file=save_file)
    if len(encodings) == 0:
        raise ValueError("No face encodings found. Check your training directory.")
    clf = train_classifier(encodings, names)
    return clf

def assign_faces_per_frame(pred_probs, class_labels, threshold=0.5):
    """
    pred_probs: list of softmax output arrays (each of shape [num_classes])
    class_labels: list of class names (length = num_classes)
    threshold: optional min confidence to accept assignment

    Returns: dict {face_index: class_name or "Unknown"}
    """
    # maximize probabilities → minimize negative
    cost_matrix = -np.array(pred_probs)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment: dict[int, Union[str, None]] = {}
    for face_idx, class_idx in zip(row_ind, col_ind):
        confidence = pred_probs[face_idx][class_idx]
        if confidence >= threshold:
            assignment[face_idx] = class_labels[class_idx]
        else:
            assignment[face_idx] = None

    return assignment
