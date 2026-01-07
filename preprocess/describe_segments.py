from collections import defaultdict
import pandas as pd
import random
import traceback
import os
from tqdm import tqdm
import numpy as np

from people_tracking.utils import get_description_from_frames

all_image_dir = '/mnt/ssd0/Images/CASTLE/'
feature_dir = '/home/allie/data_mysceal/CASTLE/siglip-so400m-patch14-384/'

# photo_features = np.load(f"{feature_dir}/features.npy")
photo_ids = pd.read_csv(f"{feature_dir}/photo_ids.csv")['photo_id']

segments = pd.read_csv('/mnt/ssd0/castle/processed/metadata_transcript_event_segmentation_new.csv')
# columns: imageID,videoID,image_path,video_path,owner,day,hour,start_time,end_time,transcript,event_id,blip2_embedding
print("Total segments:", len(segments))
boundaries = segments['image_path'].unique()
boundaries = set(boundaries)
boundaries = { path.replace('/mnt/10TBHDD/CASTLE/keyframe/', '') for path in boundaries }

segment_info = []
current_segment = []
for photo_id in photo_ids:
    if photo_id in boundaries:
        if current_segment:
            segment_info.append(current_segment)
        current_segment = [photo_id]
    else:
        current_segment.append(photo_id)

if current_segment:
    segment_info.append(current_segment)


print("Total segments found:", len(segment_info))
CATEGORIES = [
    "Eating", "Drinking", "Cooking", "Cleaning", "Washing dishes",
    "Meeting", "Leisure Activities", "Talking", "Discussing",
    "Reading", "Watching TV", "Playing Games", "Playing Guitar",
    "Walking", "Sitting", "Driving", "Using Laptop", "Using Phone",
    "Setting Table", "Serving Food", "Serving Drink",
    "Moving Furniture", "Making Tea", "Making Coffee",
]
transcript = segments['transcript']
descriptions = []
description_path = 'segment_descriptions.csv'
if os.path.exists(description_path):
    print("Loading existing descriptions from:", description_path)
    descriptions_df = pd.read_csv(description_path)
    descriptions = descriptions_df['description'].tolist()
    print("Loaded", len(descriptions), "descriptions.")

assert len(descriptions) <= len(segment_info), "More descriptions than segments!"

try:
    for i, segment in tqdm(enumerate(segment_info), total=len(segment_info)):
        restart = False
        if i < len(descriptions):
            possible_categories = [c for c in CATEGORIES if c.lower() in str(descriptions[i]).lower()]
            if len(possible_categories) == 1:
                continue
            restart = True
            print(f"Re-evaluating segment {i} with existing description: {descriptions[i]}")
            continue

        image_bytes = []
        if len(segment) > 20:
            segment = [segment[i] for i in sorted(random.sample(range(len(segment)), 20))]

        for image_path in segment:
            img = open(f"{all_image_dir}/{image_path}", 'rb').read()
            image_bytes.append(img)

        while True:
            try:
                description = get_description_from_frames([
                    f"Classify this segment into one of the following categories: {', '.join(CATEGORIES)}. "
                    f"Take into account the transcript: {transcript[i]}. Classify the segment again. Reply with only the category name with no additional text."
                    "If you are unsure, reply with 'Unclear'."
                ], image_bytes)
                if restart:
                    descriptions[i] = description
                    print(f"Updated description for segment {i}: {description}")
                else:
                    descriptions.append(description)
                break
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                traceback.print_exc()
                continue

except KeyboardInterrupt:
    print("Interrupted by user. Saving progress...")
except Exception as e:
    traceback.print_exc()
    pass

print("Total descriptions collected:", len(descriptions))
# Save the descriptions to a CSV file
descriptions_df = pd.DataFrame({
    'index': range(len(descriptions)),
    'description': descriptions
})
descriptions_df.to_csv(description_path, index=False)

# Organize segments by:
# day -> person -> list of (description, start_time, end_time, transcript)

organized_segments = {}
for i, segment in enumerate(segment_info):
    if i >= len(descriptions):
        break
    desc = descriptions[i]
    start_time = segments.iloc[i]['start_time']
    end_time = segments.iloc[i]['end_time']
    event_id = segments.iloc[i]['event_id']
    hour = segments.iloc[i]['hour']
    day = segments.iloc[i]['day']
    owner = segments.iloc[i]['owner']

    try:
        possible_categories = [c for c in CATEGORIES if c.lower() in str(desc.lower())]
    except Exception as e:
        possible_categories = []

    if day not in organized_segments:
        organized_segments[day] = {}
    if owner not in organized_segments[day]:
        organized_segments[day][owner] = []

    organized_segments[day][owner].append({
        'segment_index': i,
        'description': possible_categories[0] if possible_categories else 'Unclear',
        'start_time': start_time,
        'end_time': end_time,
        'start_seconds': int(hour) * 3600 + start_time,
        'end_seconds': int(hour) * 3600 + end_time,
        'hour': hour,
        'event_id': event_id,
    })


# Save to small files per day
all_days = list(organized_segments.keys())
all_people = set()
for day in all_days:
    for person in organized_segments[day].keys():
        all_people.add(person)

for day in all_days:
    for person in organized_segments[day].keys():
        added = 0
        day_person_segments = organized_segments[day][person]
        start_time = min([seg['start_seconds'] for seg in day_person_segments])
        # round down to nearest hour
        start_time = (start_time // 3600) * 3600
        end_time = max([seg['end_seconds'] for seg in day_person_segments])
        # round up to nearest hour
        end_time = ((end_time + 3599) // 3600) * 3600

        # Insert empty segments for gaps
        new_segments = []
        hour_covered = defaultdict(int)
        if day_person_segments[0]['start_seconds'] > start_time:
            new_segments.append({
                'segment_index': -1,
                'description': 'No Activity',
                'start_time': 0,
                'end_time': day_person_segments[0]['start_time'],
                'start_seconds': start_time,
                'end_seconds': day_person_segments[0]['start_seconds'],
                'hour': start_time // 3600,
                'event_id': -1,
            })
            added += 1
            hour_covered[start_time // 3600] += start_time

        new_segments.append(day_person_segments[0])
        for i in range(len(day_person_segments) - 1):
            j = i + 1
            if day_person_segments[i]['end_seconds'] < day_person_segments[j]['start_seconds']:
                new_segments.append({
                    'segment_index': -1,
                    'description': 'No Activity',
                    'start_time': day_person_segments[i]['end_time'],
                    'end_time': day_person_segments[j]['start_time'],
                    'start_seconds': day_person_segments[i]['end_seconds'],
                    'end_seconds': day_person_segments[j]['start_seconds'],
                    'hour': day_person_segments[i]['hour'],
                    'event_id': -1,
                })
                added += 1

            # Adjust start_time to make sure it's not less than previous end_time
            if day_person_segments[j]['start_seconds'] < day_person_segments[i]['end_seconds']:
                day_person_segments[j]['start_seconds'] = day_person_segments[i]['end_seconds']
                print("Adjusted overlapping segment for", day, person)
            new_segments.append(day_person_segments[j])

        if day_person_segments[-1]['end_seconds'] < end_time:
            new_segments.append({
                'segment_index': -1,
                'description': 'No Activity',
                'start_time': day_person_segments[-1]['end_time'],
                'end_time': end_time - (day_person_segments[-1]['hour'] * 3600),
                'start_seconds': day_person_segments[-1]['end_seconds'],
                'end_seconds': end_time,
                'hour': end_time // 3600 - 1,
                'event_id': -1,
            })
            added += 1

        print(f"Total segments for {day} {person}: {len(day_person_segments)} + {added} added = {len(new_segments)}")

        day_person_segments = new_segments
        day_person_df = pd.DataFrame(day_person_segments)
        os.makedirs(f'organized_segments/{day}', exist_ok=True)
        day_person_df.to_csv(f'organized_segments/{day}/{person}_segments.csv', index=False)


        # Use a smoothing window to visualise on the frontend (e.g 1 minute)
        timeline = []
        for hour in range(start_time // 3600, end_time // 3600):
            hour_start = hour * 3600
            hour_end = (hour + 1) * 3600
            hour_segments = [seg for seg in day_person_segments if seg['start_seconds'] <= hour_end and seg['end_seconds'] > hour_start]
            if not hour_segments:
                timeline.append({
                    'hour': hour,
                    'description': 'No Activity',
                    'start_seconds': hour_start,
                    'end_seconds': hour_end,
                })
                continue

            # Create a second-level timeline for the hour
            second_timeline = [''] * 3600
            for seg in hour_segments:
                if seg['description'] == 'No Activity':
                    continue
                seg_start = max(seg['start_seconds'], hour_start) - hour_start
                seg_end = min(seg['end_seconds'], hour_end) - hour_start
                for s in range(seg_start, seg_end):
                    second_timeline[s] = seg['description']

            # Smooth the timeline with a 1-minute window
            smoothed_timeline = []
            window_size = 5
            for s in range(0, 3600, window_size):
                window = second_timeline[s:s+window_size]
                window = [w for w in window if w]
                if window:
                    most_common = max(set(window), key=window.count)
                else:
                    most_common = 'No Activity'

                smoothed_timeline.append({
                    'hour': hour,
                    'description': most_common,
                    'start_seconds': hour_start + s,
                    'end_seconds': hour_start + s + window_size,
                })


            timeline.extend(smoothed_timeline)

        timeline_df = pd.DataFrame(timeline)
        timeline_df.to_csv(f'organized_segments/{day}/{person}_timeline.csv', index=False)
