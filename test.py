from datasets import load_dataset

ds    = load_dataset("avinot/Champion-Similarity-v6")
train = ds["train"]
val   = ds["validation"]

# Extract anchor champion names from each split
train_anchors = set(row["anchor_champion"] for row in train)
val_anchors   = set(row["anchor_champion"] for row in val)

print("Champions only in val:", val_anchors - train_anchors)
print("Champions only in train:", train_anchors - val_anchors)
print("Champions in both:", len(train_anchors & val_anchors))