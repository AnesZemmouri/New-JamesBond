# JamesBond AI

**How much does an actor's face resemble the six Eon-era James Bonds?**

This repo answers one narrow, falsifiable question: given a face, how similar is it
to the precedent Bonds — Connery, Lazenby, Moore, Dalton, Brosnan, Craig? It is a
**resemblance score**, not a casting opinion. It is **not** "who should be the next
Bond."

It ships the full pipeline plus the honesty checks that decide whether the score
means anything — and it currently reports that the score **does not** hold up. See
[Results](#results). That is the point of the project.

---

## The question, and what it is not

| | |
|---|---|
| Positive class | the 6 Eon Bonds — `isBond == 1` in `dataset/actors.csv` |
| Negative class | 30 real non-Bond actors — `isBond == 0` in `dataset/actors.csv` |
| Scored, never trained on | 12 candidates in `dataset/candidates.csv` |
| Features | 512-d InsightFace `buffalo_l` face embeddings, L2-normalised |
| Headline metric | ROC-AUC / average precision under leave-one-**actor**-out CV |

Two rules the pipeline enforces rather than documents:

1. **Candidates are scoring-only.** They are removed from training and from every CV
   fold *before* any split, as an `assert`. A model that trained on an actor cannot
   tell you anything about that actor.
2. **The unit of evaluation is the actor, not the image.** Five photos of one person
   are not five independent observations. Image-level numbers are printed too, and
   labelled *inflated*.

Read the output as a **rank / AUC**. Never as accuracy.

---

## Results

The current run is a **negative result**, and the notebook says so in its own output.

**Actor-level leave-one-actor-out, 36 actors (6 Bond / 30 negative):**

| model | ROC-AUC | AP |
|---|---|---|
| Bond-centroid cosine (training-free) | **0.783** | 0.626 |
| LogReg PCA(5), C=0.3 | 0.506 | 0.310 |

The training-free centroid is *better than* logistic regression. A model at AUC 0.506
is a coin flip.

**Robustness** — 20 repeats, 80% of negatives each: ROC-AUC mean 0.606, **std 0.076**
(range 0.458–0.750). The std is the honest error bar; the point estimate is one draw
from a wide distribution.

**Photo-source check** — a classifier given *only* image metadata (detector score,
face-box size and shape, image dimensions), no pixels, separates Bond from non-Bond
at **AUC 0.929**. The two sets were photographed differently, so part of what the face
embeddings measure is *which photoshoot the photo came from*, not the face.

**Consequence for the candidates:** no candidate reaches a meaningful cosine (≥ 0.30)
to any Bond — best is 0.167. Below that threshold "nearest Bond" is an argmax over
noise, and the notebook refuses to report it. Candidate `fit_probability` values are
listed but flagged unreliable (`actor-level CV ROC-AUC 0.506`).

To make the headline trustworthy, the classes need matching photo sources — same
kind of shot, same crop, same era of press photo — and a larger, more varied negative
class.

---

## Repository layout

```
JamesBond AI/
├── FaceModel/
│   ├── build_embeddings.py    # pictures/ -> embeddings (the data pipeline)
│   └── notebook.ipynb         # scratch: what InsightFace returns for one image
├── dataset/
│   ├── actors.csv             # 6 Bonds + 30 non-Bond negatives  (training)
│   ├── candidates.csv         # 12 actors to score, isCandidate=1 (scoring only)
│   ├── embeddings.npy         # (n_kept, 512) float32, L2-normalised
│   ├── embeddings.csv         # same values as text (e000..e511)
│   ├── images.csv             # one row per image, kept OR rejected + reason
│   ├── fetch_candidates.py    # Wikidata -> candidates.csv
│   ├── find_bond_movies.py    # filmographies -> james_bond_movies.csv
│   └── *-filmography.{csv,json}
├── pictures/<INITIALS>/*.jpg  # 48 folders: <initial> -> actor_id via actors.csv
├── classification.ipynb       # the analysis: CV, guards, figures, model fit
├── classification.html        # exported notebook
├── figures/*.png              # the 8 figures the notebook writes
├── model.joblib               # fitted model + its metadata
├── main.py                    # placeholder ("Hello from jamesbond-ai!")
└── pyproject.toml / uv.lock   # deps, Python 3.13
```

`_patch2.py` is a one-off maintenance helper that rewrites guard messages inside
`classification.ipynb`; it is not part of the pipeline.

---

## Setup

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/). Dependencies:
`insightface`, `scikit-learn`, `pandas`, `matplotlib`, `ipykernel`.

```bash
uv sync
```

`insightface` downloads the `buffalo_l` model pack on first use (needs network).
GPU is used when available (`ctx_id=0`); pass `--ctx-id -1` for CPU.

---

## Usage

### 1. Add face photos

One folder per actor under `pictures/`, named by either `actor_id`
(`daniel-craig`) or the `initials` column (`DC`). Five or more photos per actor —
a mean over one or two photos is that photo, not that actor.

```
pictures/DC/007.jpg
pictures/DC/DC2.jpg
```

The folder must map to a row in `dataset/actors.csv` or `dataset/candidates.csv`,
otherwise the script warns and uses the folder name verbatim (the actor is then in
neither class and will fail a later assertion).

### 2. Build embeddings

```bash
uv run FaceModel/build_embeddings.py
# options: --min-score 0.5 --min-face-px 80 --ctx-id 0 --emb-format both
```

Per image it keeps the **largest** face passing the quality gates, and writes
`images.csv`, `embeddings.npy` and `embeddings.csv`. Quality gates and flags:

| gate / flag | meaning |
|---|---|
| `no_face`, `low_score`, `small_face`, `unreadable` | rejected, reason recorded |
| `multi_face` | group shot — dropped later (the largest face may not be the actor) |
| `outlier` | mean cosine to its siblings far below the actor's median |

### 3. Run the analysis

Open `classification.ipynb` and run all. It asserts its way through the data
(labels aligned, no candidate in training, real negative class present) before
reporting anything, writes the 8 figures to `figures/`, and fits `model.joblib`.

### 4. Utilities

```bash
uv run dataset/fetch_candidates.py --min-year 1980 --max-year 1996 --min-sitelinks 5
uv run dataset/find_bond_movies.py --only-found
```

`fetch_candidates.py` queries Wikidata for UK/Irish male actors in a birth-year
window (a *nationality* filter mirroring the real casting constraint) and writes
`candidates.csv`. `find_bond_movies.py` matches a canonical list — 25 Eon films plus
2 non-Eon (1967 *Casino Royale*, *Never Say Never Again*) — against the
Letterboxd-sourced `*_filmography.csv` files by slug — deliberately not a
"grep for bond" pass, which would also catch ~40 documentaries and TV specials.

---

## Guard rails the notebook enforces

These fail loudly rather than returning a plausible-looking number.

- **A real negative class is required** (`MIN_NEG_ACTORS = 25`, `MIN_NEG_IMAGES = 5`).
  The candidates are the people being *scored*. Using them as negatives trains the
  model to answer the very question it is being asked, so every candidate comes back
  "not Bond" and the ranking is meaningless.
- **Candidates never leak into training**, checked by `assert`, not by convention.
- **`HOLDOUT_ACTOR` is a candidate**, so it is held out automatically.
- **A `k`/`C` sweep is a diagnostic, not a selector.** The shipped model uses the
  fixed `K_PCA = 5`, `C_LOGREG = 0.3` from the top of the notebook. The sweep's
  spread across `k` is 0.454–0.877 — a larger range than the gap between any two
  models in it, which is the notebook's evidence that picking the best cell is
  picking noise.

---

## Known limitations

- **Photo source confound.** Metadata alone reaches AUC 0.929 — the classes are
  photographed differently. Fix the data before trusting any face-level number.
- **The six Bonds span three eras of film stock and press photography**, so a single
  centroid over them is a rough target at best.
- **`model.joblib` is not portable as-is.** The pipeline pickles `ClampedPCA`, which
  is defined inside the notebook (`__main__`), so `joblib.load` outside a run of that
  notebook raises `AttributeError`. Loading it requires that class to be importable.
- **Small sample.** 36 training actors — 6 positives — makes actor-level AUC a coarse
  diagnostic, not a precise estimate.

---

## Figures

Written to `figures/` at 150 dpi, matplotlib only.

| figure | what it answers |
|---|---|
| `pca_face_space.png` | where the actors sit relative to each other |
| `auc_vs_k.png` | how much the result moves with the number of PCA components |
| `actor_scores.png` | one bar per training actor, sorted |
| `actor_roc.png` | actor-level ROC, model vs baseline |
| `repeated_auc.png` | the spread from the robustness section |
| `candidate_bond_heatmap.png` | candidate × Bond cosine similarity |
| `candidate_ranking.png` | the ranking, with the CV AUC in the title |
| `photo_source.png` | whether the photos themselves give the label away |
