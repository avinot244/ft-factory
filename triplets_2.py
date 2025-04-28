# triplet_miner.py

import random
import json

# 1. Champion Metadata (example with partial data — you should extend this)
champion_metadata = {
    "Yasuo": {"class": "Fighter", "position": "Mid"},
    "Yone": {"class": "Fighter", "position": "Mid"},
    "Ahri": {"class": "Mage", "position": "Mid"},
    "Zed": {"class": "Assassin", "position": "Mid"},
    "Thresh": {"class": "Tank", "position": "Support"},
    "Blitzcrank": {"class": "Tank", "position": "Support"},
    "Jhin": {"class": "Marksman", "position": "ADC"},
    "Caitlyn": {"class": "Marksman", "position": "ADC"},
    "Soraka": {"class": "Support", "position": "Support"},
    "Amumu": {"class": "Tank", "position": "Jungle"},
    "Lee Sin": {"class": "Fighter", "position": "Jungle"},
    "Jarvan IV": {"class": "Tank", "position": "Jungle"},
    "Vayne": {"class": "Marksman", "position": "ADC"},
    "Veigar": {"class": "Mage", "position": "Mid"},
    "Morgana": {"class": "Mage", "position": "Support"},
    "Malphite": {"class": "Tank", "position": "Top"},
    "Janna": {"class": "Support", "position": "Support"},
    "Darius": {"class": "Fighter", "position": "Top"},
    # 🛑 You should extend this to full champion pool (~160 champions) for best results!
}

# Extract Champion List
champions = list(champion_metadata.keys())

# 2. Auto-generate triplets
triplets = []

def sample_positive(anchor, anchor_info):
    # Find champions with same class or same position
    candidates = [
        champ for champ, info in champion_metadata.items()
        if champ != anchor and (info["class"] == anchor_info["class"] or info["position"] == anchor_info["position"])
    ]
    return random.choice(candidates) if candidates else None

def sample_negative(anchor, anchor_info):
    # Find champions with different class and position
    candidates = [
        champ for champ, info in champion_metadata.items()
        if champ != anchor and (info["class"] != anchor_info["class"] and info["position"] != anchor_info["position"])
    ]
    return random.choice(candidates) if candidates else None

num_triplets_target = 10000

for _ in range(num_triplets_target):
    anchor = random.choice(champions)
    anchor_info = champion_metadata[anchor]

    positive = sample_positive(anchor, anchor_info)
    negative = sample_negative(anchor, anchor_info)

    if positive and negative:
        triplets.append((anchor, positive, negative))

print(f"Generated {len(triplets)} triplets!")

# 3. Save to disk
with open("lol_champion_triplets.jsonl", "w") as f:
    for anchor, positive, negative in triplets:
        json.dump({"anchor": anchor, "positive": positive, "negative": negative}, f)
        f.write("\n")

print("Triplets saved to lol_champion_triplets.jsonl ✅")
