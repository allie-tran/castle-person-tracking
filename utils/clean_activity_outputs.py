import pandas as pd

df = pd.read_csv('segment_descriptions.csv')

# index, description

descriptions = df['description'].tolist()
new_descriptions = []

CATEGORIES = [
    "Eating", "Drinking", "Cooking", "Cleaning", "Washing dishes",
    "Meeting", "Leisure Activities", "Talking", "Discussing",
    "Reading", "Watching TV", "Playing Games", "Playing Guitar",
    "Walking", "Sitting", "Driving", "Using Laptop", "Using Phone",
    "Setting Table", "Serving Food", "Serving Drink",
    "Moving Furniture", "Making Tea", "Making Coffee",
]

multiples = []
no_categories = []

for i, desc in enumerate(descriptions):
    desc = str(desc).strip()


    # get the categories that match the description, from the order of appearance in the description
    possible_categories = [cat for cat in CATEGORIES if cat.lower() in desc.lower()]
    possible_categories = list(set(possible_categories))  # Remove duplicates
    possible_categories.sort(key=lambda x: desc.lower().index(x.lower()) if x.lower() in desc.lower() else float('inf'))

    if len(possible_categories) == 1:
        new_desc = possible_categories[0]
    elif len(possible_categories) > 1:
        print(f"Multiple categories found for description {i}: {desc}")
        new_desc = possible_categories[0]  # Default to the first match
        multiples.append((i, desc, possible_categories))
    else:
        print(f"No category found for description {i}: {desc}")
        new_desc = "Uncategorized"
        no_categories.append((i, desc))
    new_descriptions.append(new_desc)

df['description'] = new_descriptions
print(f"Total descriptions: {len(descriptions)}")
print(f"Total multiples: {len(multiples)}")
print(f"Total uncategorized: {len(no_categories)}")

for i, desc, cats in multiples:
    print(desc)
    print("--" * 20)
    print(f"Possible categories: {', '.join(cats)}")
    input("Press Enter to continue...")
    print("--" * 20)
