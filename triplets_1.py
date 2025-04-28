import random
import pandas as pd
from collections import defaultdict

# Example champion data
champions = [
    {"name": "Aatrox", "classes": ["Juggernaut"], "roles": ["Top"]},
    {"name": "Ahri", "classes": ["Burst"], "roles": ["Mid"]},
    {"name": "Akali", "classes": ["Assassin"], "roles": ["Mid", "Top"]},
    {"name": "Akshan", "classes": ["Marksman", "Assassin"], "roles": ["Mid", "Top"]},
    {"name": "Alistar", "classes": ["Vanguard"], "roles": ["Support"]},
    # ... (complete your full champion list here)
]

# Group champions by class
class_to_champions = defaultdict(list)
for champ in champions:
    for c in champ["classes"]:
        class_to_champions[c].append(champ)

# Group champions by role
role_to_champions = defaultdict(list)
for champ in champions:
    for r in champ["roles"]:
        role_to_champions[r].append(champ)

def generate_triplets(champions, num_triplets_per_champion=5):
    triplets = []
    
    for anchor in champions:
        # Try finding positives: champions sharing at least one class
        positive_candidates = set()
        for c in anchor["classes"]:
            positive_candidates.update(class_to_champions[c])
        positive_candidates.discard(anchor)  # remove anchor itself
        
        # Try finding negatives: champions with no class overlap
        negative_candidates = [c for c in champions if set(c["classes"]).isdisjoint(anchor["classes"])]
        
        # Skip if not enough positives or negatives
        if not positive_candidates or not negative_candidates:
            continue
        
        for _ in range(num_triplets_per_champion):
            positive = random.choice(list(positive_candidates))
            negative = random.choice(negative_candidates)
            triplets.append((anchor["name"], positive["name"], negative["name"]))
    
    return triplets

# Generate triplets
triplets = generate_triplets(champions, num_triplets_per_champion=5)

# Example output
for anchor, positive, negative in triplets[:10]:
    print(f"Anchor: {anchor}, Positive: {positive}, Negative: {negative}")

# Optional: save to CSV
df_triplets = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])
df_triplets.to_csv("triplets.csv", index=False)
