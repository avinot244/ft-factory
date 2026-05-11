import json
import re

FORBIDDEN = re.compile(
    r"\b(tank|bruiser|assassin|mage|marksman|ADC|carry|support|"
    r"jungler|top laner|mid laner|juggernaut|skirmisher|diver|"
    r"enchanter|controller|engage|peel|frontline|backline|"
    r"excels at|relies on|focuses on|kit includes|provides|"
    r"deals damage)\b",
    re.IGNORECASE
)

with open("./data/champion_rationales.jsonl") as f:
    rationales = [json.loads(line) for line in f if line.strip()]

violations = {}
for r in rationales:
    hits = FORBIDDEN.findall(r["rationale"])
    if hits:
        violations[r["name"]] = list(set(hits))

print(f"Champions with forbidden vocabulary: {len(violations)}/{len(rationales)}")
for name, hits in list(violations.items())[:15]:
    print(f"  {name}: {hits}")

# Also check axis labels leaked into output
AXIS_LABELS = re.compile(
    r"\[(EXECUTION PATTERN|DECISION FREQUENCY|SKILL TRANSFER|FAILURE MODE)\]"
)
axis_leaks = [r["name"] for r in rationales
              if AXIS_LABELS.search(r["rationale"])]
print(f"\nAxis label leaks: {len(axis_leaks)} champions")
if axis_leaks:
    print(f"  {axis_leaks[:10]}")