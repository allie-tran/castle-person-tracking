# You can install it using pip:
# pip install ultralytics

import json
import os
import glob
from typing import Any
from tqdm import tqdm


import cv2
from ultralytics import YOLO


def extract_objects_from_images(
    input_dir, output_dir="extracted_objects_from_images"
):
    """
    Extracts objects from image files in a directory (including subfolders)
    using a YOLOv11 model.

    Args:
        input_dir (str): Path to the input directory containing images and subfolders.
        model_name (str): Name of the YOLOv11 model to use (e.g., 'yolov11n.pt', 'yolov11s.pt').
                          'n' stands for nano, 's' for small.
                          You can find other models on the Ultralytics GitHub or documentation.
        output_dir (str): Directory to save extracted object images.
    """
    print(f"Loading YOLOv11 models")
    try:
        # Load a pre-trained YOLOv11 model
        # The model will be downloaded automatically if not found locally.
        detect_model = YOLO('yolo11x.pt', task='detect', verbose=False)
        classify_model = YOLO('yolo11x-cls.pt', task='classify', verbose=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Please ensure you have an active internet connection for initial download or check model name."
        )
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_count = 0
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    print(f"Searching for images in: {input_dir} and its subfolders...")

    # Walk through the input directory and its subfolders
    output_path = os.path.join(output_dir, "objects_with_classification.json")
    if not os.path.exists(output_path):
        output: dict[str, list[dict[str, Any]]] = {}
    else:
        with open(output_path, "r") as f:
            output = json.load(f)

    try:
        image_list = glob.glob(os.path.join(input_dir, "**", "10_*.*"), recursive=True)
        image_list = [
            img for img in image_list if img.lower().endswith(supported_extensions)
        ]
        for image_path in tqdm(image_list, desc="Processing images"):
            key = os.path.relpath(image_path, input_dir)  # Relative path for output

            if key in output:
                continue  # Skip if already processed

            image_count += 1

            # Read the image
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # Perform object detection on the image
            # The 'conf' argument sets the confidence threshold for detections.
            # The 'iou' argument sets the Intersection Over Union threshold for non-maximum suppression.
            results = detect_model(
                frame, verbose=False
            )  # Adjust confidence and iou as needed
            # Initialize the output for this image
            output[key] = []

            # Iterate through the detection results
            for r in results:
                # 'boxes' contains bounding box coordinates, class IDs, and confidence scores
                # 'xyxy' format: [x1, y1, x2, y2]
                boxes = r.boxes

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0]
                    )  # Get integer coordinates
                    conf = box.conf[0]  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    class_name = detect_model.names[cls]  # Get class name from model

                    # Extract the detected object from the frame
                    # Ensure coordinates are within frame boundaries
                    h, w, _ = frame.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Check if the bounding box is valid (positive dimensions)
                    if x2 > x1 and y2 > y1 and class_name == "person":
                        output_file = os.path.join(
                            output_dir, f"{key.replace('/', '_')}_{i}.jpg"
                        )
                        # Save the extracted object image
                        img = frame[y1:y2, x1:x2]
                        cv2.imwrite(output_file, img)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        pass
    except Exception as e:
        print(f"Error processing images: {e}")
        pass

    print(f"Extracted {len(output)} objects from {image_count} images.")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Object extraction complete. Extracted objects saved to: {output_dir}")
    print(f"Processed {image_count} images.")


# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/image_directory' with the actual path
    # to the directory containing your images and subfolders.
    input_image_directory = "/mnt/ssd0/Images/CASTLE_jpg/"
    # Call the function to start object extraction from images
    extract_objects_from_images(
        input_image_directory,
        output_dir="extracted_objects_from_images"
    )
