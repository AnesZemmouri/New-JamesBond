"""Point the guard/demo instructions at candidates.csv as well as actors.csv."""
from pathlib import Path

import nbformat as nbf

NB = Path("classification.ipynb")
nb = nbf.read(NB, as_version=4)

REPLACEMENTS = [
    (
        '"  1. add non-Bond actors to dataset/actors.csv with isBond = 0 "',
        '"  1. add non-Bond actors to dataset/candidates.csv (or actors.csv) "\n        "with isBond = 0 "',
    ),
    (
        '"  1. dataset/actors.csv  -> a row with actor_id=tom-holland, isBond=0, heightCm=...\\n"',
        '"  1. dataset/candidates.csv -> a row with actor_id=tom-holland, isBond=0, heightCm=...\\n"',
    ),
]

counts = dict.fromkeys(range(len(REPLACEMENTS)), 0)
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    for i, (old, new) in enumerate(REPLACEMENTS):
        if old in cell.source:
            cell.source = cell.source.replace(old, new)
            counts[i] += 1

for i, n in counts.items():
    if n != 1:
        raise SystemExit(f"replacement {i} applied {n} times, expected 1")

nbf.write(nb, NB)
nbf.validate(nb)
print("patched + validated:", NB)
