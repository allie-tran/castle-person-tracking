from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
import torch


class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()  # type: ignore
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract features from a list of RGB crops (as np arrays)."""
        images = [Image.fromarray(crop) for crop in crops]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(  # type: ignore
            self.device
        )
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)  # type: ignore
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        return features.cpu().numpy()


# extractor = utils.FeatureExtractor(model_name="osnet_x1_0", device="cuda")
extractor = CLIPFeatureExtractor(
    model_name="openai/clip-vit-base-patch32", device="cuda"
)


def extract_person_features(crops: list[np.ndarray]) -> np.ndarray:
    """
    Extract features from a given crop using the feature extractor.

    Args:
            crop (np.ndarray): The image crop to extract features from.

    Returns:
            torch.Tensor: The extracted features.
    """
    # Extract features
    features = extractor(crops)
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    return features

# def cluster_track_ids(features: np.ndarray, track_ids: list[int]) -> dict[int, set]:
#     features = features.reshape(features.shape[0], -1)  # Ensure 2D shape
#     dbscan = DBSCAN(eps=0.1, min_samples=5)
#     dbscan.fit(features)
#     labels = dbscan.labels_

#     clustered_track_ids = defaultdict(set)
#     own_track_id = len(set(labels))  # Unique ID for noise points
#     for track_id, label in zip(track_ids, labels):
#         if label == -1:  # noise points
#             clustered_track_ids[own_track_id].add(track_id)
#             own_track_id += 1
#         clustered_track_ids[label].add(track_id)
#     return clustered_track_ids


def track_center(bbox: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def should_merge(
    track_a_frames: list[int],
    track_b_frames: list[int],
    track_a_bboxes: list[np.ndarray],
    track_b_bboxes: list[np.ndarray],
    track_a_frame_ids: set[int],
    track_b_frame_ids: set[int],
    track_a_features: np.ndarray,
    track_b_features: np.ndarray,
) -> bool:
    if track_a_frame_ids & track_b_frame_ids:
        return False

    latest_a = max(track_a_frames)
    earliest_b = min(track_b_frames)

    if earliest_b < latest_a:
        return False
    if earliest_b - latest_a > 20:
        return False

    fa = track_a_features
    fb = track_b_features
    similarity = cosine_similarity(fa.reshape(1, -1), fb.reshape(1, -1))[0][0]
    if similarity < 0.5:
        return False

    last_box = track_a_bboxes[-1]
    first_box = track_b_bboxes[0]
    center_a = track_center(last_box)
    center_b = track_center(first_box)
    distance = np.linalg.norm(np.array(center_a) - np.array(center_b))
    if distance > 100:
        return False

    return True


def merge_track_ids(
    track_id_to_features: dict[int, np.ndarray],
    track_id_to_frames: dict[int, list[int]],
    track_id_to_bboxes: dict[int, list[np.ndarray]],
):
    merge_track_ids = defaultdict(set)
    track_ids = list(track_id_to_features.keys())
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            track_a = track_ids[i]
            track_b = track_ids[j]

            if should_merge(
                track_id_to_frames[track_a],
                track_id_to_frames[track_b],
                track_id_to_bboxes[track_a],
                track_id_to_bboxes[track_b],
                set(track_id_to_frames[track_a]),
                set(track_id_to_frames[track_b]),
                track_id_to_features[track_a],
                track_id_to_features[track_b],
            ):
                merge_track_ids[track_a].add(track_b)
                merge_track_ids[track_b].add(track_a)

    # add the track itself to the merge set
    for track_id in track_ids:
        merge_track_ids[track_id].add(track_id)
    return merge_track_ids
