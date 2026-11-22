"""One-off: write eval/dev_split.json, a type-stratified ~30% tuning split (seed 0).

Tune on dev; report on test (everything else). Do not regenerate after tuning starts.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

root = Path(__file__).parent
qs = json.loads((root / "questions.json").read_text(encoding="utf-8"))["questions"]
by_type = defaultdict(list)
for q in qs:
    by_type[q["type"]].append(q["id"])
rng = random.Random(0)
ids = []
for t, group in sorted(by_type.items()):
    ids += rng.sample(group, max(2, round(0.3 * len(group))))
(root / "dev_split.json").write_text(
    json.dumps({"seed": 0, "ids": sorted(ids)}, indent=1), encoding="utf-8"
)
print(len(ids), {t: max(2, round(0.3 * len(g))) for t, g in sorted(by_type.items())})
