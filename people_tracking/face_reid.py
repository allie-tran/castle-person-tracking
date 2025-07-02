import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Union

def assign_faces_per_frame(pred_probs, class_labels, threshold=0.5):
    """
    pred_probs: list of softmax output arrays (each of shape [num_classes])
    class_labels: list of class names (length = num_classes)
    threshold: optional min confidence to accept assignment

    Returns: dict {face_index: class_name or "Unknown"}
    """
    cost_matrix = -np.array(pred_probs)  # maximize probabilities → minimize negative

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
