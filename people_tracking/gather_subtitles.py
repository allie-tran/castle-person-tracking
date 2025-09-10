# This script will gather subtitles from a given segment
import json
import os

from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR", ".")


def get_timestamp_from_image_key(image_key: str, fps: int = 2) -> int:
    """
    Convert image_key to timestamp
    :param image_key: day/name/hour_frameid
    :return: timestamp in format "day_name_hour_frameid"
    """
    image_key = image_key.split(".")[0]
    parts = image_key.split("/")
    hour, frame_id = parts[-1].split("_")
    hour = int(hour)
    frame_id = int(frame_id)
    seconds = frame_id // fps
    return hour * 3600 + seconds

def get_subtitles_for_day(day: str, person: str):
    subtitles = []
    for hour in range(24):
        path = f"{ROOT_DIR}/{day}/{person}/transcript/{hour}.json"
        if os.path.exists(path):
            with open(path) as f:
                transcript = json.load(f)
            transcript = transcript["chunks"]
            for chunk in transcript:
                start, end = chunk["timestamp"]
                start = int(start) if start else 0
                end = int(end) if end else 3600
                text = chunk["text"]
                subtitles.append(
                    {
                        "start": start + hour * 3600,
                        "end": end + hour * 3600,
                        "text": text,
                    }
                )
    return subtitles

def format_timestamp(timestamp: int) -> str:
    """
    Format timestamp in seconds to HH:MM:SS
    :param timestamp: timestamp in seconds
    :return: formatted timestamp
    """
    hours = timestamp // 3600
    minutes = (timestamp % 3600) // 60
    seconds = timestamp % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def gather_subtitles(segment: list[str], subtitles: list[dict], fps=2) -> str:
    """
    Gather subtitles from a given segment
    :param segment: list of image_key (day/name/hour_frameid)
    """
    start = get_timestamp_from_image_key(segment[0], fps)
    end = get_timestamp_from_image_key(segment[-1], fps)

    chunks = []
    for subtitle in subtitles:
        if subtitle["start"] <= end and subtitle["end"] >= start:
            start_time = format_timestamp(subtitle["start"])
            end_time = format_timestamp(subtitle["end"])
            chunks.append(f"[{start_time} - {end_time}]: {subtitle['text']}")

    if not chunks:
        return ""

    return "\n".join(chunks)
