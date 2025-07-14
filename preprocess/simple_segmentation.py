import json
import os

import numpy as np
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import cv2
from tqdm import tqdm

load_dotenv()
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
KEYFRAMES_DIR = os.getenv("KEYFRAMES_DIR")
assert PROCESSED_DIR is not None, "PROCESSED_DIR is not set in .env file"
assert KEYFRAMES_DIR is not None, "KEYFRAMES_DIR is not set in .env file"
MODEL = "facebook/dinov2-with-registers-small"

features = np.load(f"{PROCESSED_DIR}/features/{MODEL}/features.npy")
photo_ids = pd.read_csv(f"{PROCESSED_DIR}/features/{MODEL}/photo_ids.csv")["photo_id"]

# the photo_id has the format "day3/Allie/[hour]_[seconds].webp"
# so we need a special sort order
def name_to_sort_id(photo_id):
    """
    Convert photo_id to seconds
    """
    day, name, time = photo_id.split("/")
    hour, seconds = time.split("_")
    hour = int(hour)
    seconds = int(seconds.split(".")[0])
    all_seconds = hour * 3600 + seconds
    # there are 24 * 3600 seconds = 86400 seconds in a day
    # so 5 digits should be enough
    # if we pad with zeros, we can sort them correctly
    return f"{day}/{name}/{all_seconds:05d}"

times = [name_to_sort_id(photo_id) for photo_id in photo_ids]
sort_idx = np.argsort(times)
photo_ids = np.array(photo_ids)[sort_idx]
features = features[sort_idx]

# get normalized features
features = features / np.linalg.norm(features, axis=1, keepdims=True)

# seperate video ids
def to_video_id(photo_id):
    "day3/Allie/[hour]_[seconds].webp"
    day, person, _ = photo_id.split("/")
    return f"{day}/{person}"

video_ids = [to_video_id(photo_id) for photo_id in photo_ids]
unique_video_ids = np.unique(video_ids)
print(f"Number of unique video ids: {len(unique_video_ids)}")
print(unique_video_ids)
input("Press Enter to continue...")
video_ids = np.array(video_ids)

def frames_to_video(frames, name):
    """
    Convert frames to video
    """
    # create a video from the frames
    # use ffmpeg to create a video from the frames
    # ffmpeg -framerate 5 -i %05d.jpg -c:v libx264 -pix_fmt yuv420p out.mp4
    if len(frames) == 0:
        print(f"No frames to create video {name}.mp4")
        return
    print(f"Creating video {name}.mp4")
    # get the first frame to get the size
    if "webp" in frames[0]:
        first_frame = Image.open(frames[0])
        first_frame = np.array(first_frame.convert("RGB"))
    else:
        first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = 5
    out = cv2.VideoWriter(f"{name}.mp4", fourcc, fps * 5, (width, height))
    for frame in tqdm(frames):
        img = cv2.imread(frame)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


static_cameras = ["Kitchen", "Living1", "Living2", "Meeting", "Reading"]
THRESHOLD = 0.33
# THRESHOLD = 0.1
STATIC_THRESHOLD = 0.5 # higher means more segments

all_segments = {}
for video_id in unique_video_ids:
    # if "Kitchen" in video_id:
    #     continue
    print(f"Processing video {video_id}")
    ids = np.where(video_ids == video_id)[0]
    video_features = features[ids]
    video_frames = photo_ids[ids]

    # go through each one to find the boundaries
    boundaries = []
    anchor = 0
    scores = [(0, 0)]
    thresh = THRESHOLD
    if video_id.split("/")[1] in static_cameras:
        thresh = STATIC_THRESHOLD
    for i in range(1, len(video_features)):
        # get the distance between the two features
        sim_to_anchor = video_features[i] @ video_features[anchor]
        thresh = THRESHOLD
        if video_id.split("/")[1] in static_cameras:
            thresh = STATIC_THRESHOLD
        segment_feat = np.mean(video_features[anchor:i], axis=0)
        sim_to_segment = video_features[i] @ segment_feat
        sim = (sim_to_anchor + sim_to_segment) / 2
        if sim < thresh:
            boundaries.append(i)
            anchor = i
            scores.append((sim_to_anchor, sim_to_segment))

    # add the last one
    boundaries.append(len(video_features))
    scores.append((0, 0))

    # create segments
    segments: list[list[str]] = []
    start = 0
    for boundary in boundaries:
        segments.append([str(video_frames[i]) for i in range(start, boundary)])
        start = boundary
    # add the last segment
    segments.append([str(video_frames[i]) for i in range(start, len(video_frames))])
    print("Number of segments: ", len(segments))

    CREATE_HTML = False
    if CREATE_HTML:

        # create a html file to visualise the boundaries
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Boundaries</title>
            <style>
                .boundary {
                    position: absolute;
                    background-color: red;
                    opacity: 0.5;
                }
            </style>
        </head>
        <style>
        .image-container {
            position: relative;
            display: inline-block;
            margin: 10px;
        }
        .image-container img {
            width: 300px;
            height: auto;
        }
        .image-container p {
            position: absolute;
            bottom: 0;
            left: 0;
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            padding: 5px;
        }
        .video-container {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .segment-container {
            width: 100%;
            display: flex;
            flex-direction: row;
            overflow-x: auto;
        }
        </style>
        <body>
            <h1>Simple Boundaries</h1>
            <div class="video-container">
        """

        html += f"<h2>{video_id}</h2>"
        html += f"""
    <p>Number of segments: {len(segments)}</p>
    <p>Number of frames: {len(video_frames)}</p>
    <p>Average segment length: {len(video_frames) // len(segments)} frames</p>
    """
        for i, segment in enumerate(segments):
            html += f"<div style='height: 1px; background-color: gray; width: 100%'></div>"
            html += f"<p>Score to anchor: {scores[i][0]:.2f}</p>"
            html += f"<p>Score to segment: {scores[i][1]:.2f}</p>"
            html += f'<div class="segment-container" key="segment-{i}">'
            for image in segment:
                tag = image.split("_")[-1].split(".")[0]
                tag = int(tag) // 5
                minute = tag // 60
                second = tag % 60
                tag = f"{minute:02}:{second:02}"
                path = f"{KEYFRAMES_DIR}/{image}"
                html += f"""
                <div class="image-container">
                    <img src="{path}">
                    <p>{tag}</p>
                </div>
                """

            html += f"""
            </div>
            <p>Duration: {len(segment) // 5} seconds</p>
            """

        html += """
        </div>
        </body>
        </html>
        """

        # save the html file
        with open(f"html/{video_id.replace('/', '_')}.html", "w") as f:
            f.write(html)

        # save each segment as a video
        video_dir = f"{PROCESSED_DIR}/webp_videos/{video_id}"
        print(f"Saving {len(segments)} videos to {video_dir}")
        os.makedirs(video_dir, exist_ok=True)
        for i, segment in enumerate(segments):
            segment_file = os.path.join(video_dir, f"segment_{i}")
            paths = [f"{KEYFRAMES_DIR}/{image}" for image in segment]
            frames_to_video(paths, segment_file)

    all_segments[str(video_id)] = segments

# save segments to a json file
with open(f"{PROCESSED_DIR}/webp_segments.json", "w") as f:
    json.dump(all_segments, f, indent=4)
