import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client = MongoClient("localhost", 27017)
db = client["castle"]
transcripts_col = db["transcripts"]
try:
    transcripts_col.create_index([("person", 1), ("day", 1), ("start", 1)], unique=False)
except Exception as e:
    print(f"Index creation failed: {e}")

ROOT_DIR = "/mnt/castle/castle_downloader/CASTLE2024/main"


# === MAIN FUNCTIONALITY ===
def get_subtitles_for_day(day: str, person: str) -> list[dict[str, Any]]:
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
                subtitles.append({
                    "start": start + hour * 3600,
                    "end": end + hour * 3600,
                    "text": text
                })
    return subtitles

def index_transcripts_for_day(day: str, person: str):
    subtitles = get_subtitles_for_day(day, person)
    if not subtitles:
        print(f"No data for {person} on {day}")
        return

    for s in subtitles:
        s["person"] = person
        s["day"] = day

    transcripts_col.delete_many({"person": person, "day": day})  # Clear old entries
    transcripts_col.insert_many(subtitles)
    print(f"Indexed {len(subtitles)} entries for {person} on {day}")

def index_transcript():
    transcripts_col.drop()  # Clear the collection for fresh indexing
    # create index if it doesn't exist
    transcripts_col.create_index([("person", 1), ("day", 1), ("start", 1)], unique=False)
    people = os.listdir(f"{ROOT_DIR}/day4")  # Update to your day logic or loop over days
    days = ["day1", "day2", "day3", "day4"]
    pbar = tqdm(total=len(people) * len(days), desc="Indexing transcripts")
    for person in people:
        for day in days:
            pbar.set_description(f"Indexing {person} on {day}")
            transcript_path = f"{ROOT_DIR}/{day}/{person}/transcript"
            if os.path.isdir(transcript_path):
                index_transcripts_for_day(day, person)
            pbar.update(1)


def embed_and_store_vectors():
    model_name = "all-MiniLM-L6-v2"  # Use a better multilingual model if needed
    model = SentenceTransformer(model_name)
    cursor = transcripts_col.find()
    texts = []
    ids = []
    embeddings = []

    count = transcripts_col.count_documents({})
    print(f"Embedding {count} transcripts using model: {model_name}")

    for doc in tqdm(cursor, total=count, desc="Embedding transcripts"):
        ids.append(f"{doc['person']}/{doc['day']}/{doc['start']}:{doc['end']}")
        texts.append(doc["text"])
        emb = model.encode(doc["text"]).tolist()
        embeddings.append(emb)

    # Save
    OUTPUT_DIR = f"/home/allie/data_mysceal/CASTLE/transcripts-{model_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    embeddings = np.array(embeddings).astype(np.float32).reshape(count, -1)
    np.save(f"{OUTPUT_DIR}/features.npy", embeddings)
    pd.DataFrame({"_id": ids, "text": texts}).to_csv(f"{OUTPUT_DIR}/metadata.csv", index=False)

    print(f"✅ Embedded {count} transcripts.")


if __name__ == "__main__":
    # index_transcript()
    embed_and_store_vectors()
