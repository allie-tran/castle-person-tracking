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
    })


# Save to small files per day
all_days = list(organized_segments.keys())
all_people = set()
for day in all_days:
    for person in organized_segments[day].keys():
        all_people.add(person)

for day in all_days:
    for person in organized_segments[day].keys():
        day_person_segments = organized_segments[day][person]
        day_person_df = pd.DataFrame(day_person_segments)
        os.makedirs(f'organized_segments/{day}', exist_ok=True)
        day_person_df.to_csv(f'organized_segments/{day}/{person}_segments.csv', index=False)
