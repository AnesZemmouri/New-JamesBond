#!/usr/bin/env python
"""Extract face embeddings for every actor image folder.

Expected layout
---------------
    pictures/<actor_id>/<anything>.jpg

Each subfolder of pictures/ is one actor; <actor_id> should match actor_id in
dataset/actors.csv or dataset/candidates.csv.

Outputs (in dataset/)
---------------------
    images.csv       one row per image, kept OR rejected, with the reason
    embeddings.npy   (n_kept, 512) float32, L2-normalised (== normed_embedding)
    embeddings.csv   same values as text: actor_id, image_id, e000..e511
                     -- readable in any editor, loads with pd.read_csv

Quality gates applied per image:
    * exactly the largest face is taken, but a group shot is flagged
    * det_score must be >= --min-score
    * face box must be >= --min-face-px on its shorter side

Intra-actor consistency: for each actor the kept embeddings are compared to each
other.  An image whose mean cosine similarity to its siblings is far below the
rest is flagged \"outlier\" -- i.e. possibly the wrong person, or a very
different pose/age.

Usage
-----
    python build_embeddings.py
    python build_embeddings.py --images-dir ../pictures --min-score 0.5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_COLUMNS = [
    "image_id", "actor_id", "path", "det_score", "bbox", "n_faces",
    "embedding_row", "status", "reason", "qc_flag",
]
# If an image's mean similarity to its siblings is below the actor's median by
# more than this margin, flag it.
OUTLIER_MARGIN = 0.25


def pick_main_face(faces, min_score: float, min_px: int):
    """Largest face passing the gates, plus the reason a better one is missing."""
    if not faces:
        return None, "no_face", 0
    scored = [f for f in faces if f.det_score >= min_score]
    if not scored:
        return None, "low_score", len(faces)
    big = [f for f in scored
           if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= min_px]
    if not big:
        return None, "small_face", len(faces)
    best = max(big, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best, "", len(faces)


def load_app(ctx_id: int, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def load_id_map(root: Path) -> dict[str, str]:
    """Map folder names -> actor_id.

    Folders may be named either by actor_id (daniel-craig) or by the `initials`
    column (DC).  Both resolve, as long as the CSV carries the mapping.
    """
    mapping: dict[str, str] = {}
    for name in ("actors.csv", "candidates.csv"):
        path = root / "dataset" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                aid = (r.get("actor_id") or "").strip()
                ini = (r.get("initials") or "").strip()
                if aid:
                    mapping.setdefault(aid.lower(), aid)
                if aid and ini:
                    mapping.setdefault(ini.lower(), aid)
    return mapping


def resolve_actor_id(folder: str, id_map: dict[str, str]) -> str:
    return id_map.get(folder.lower(), folder)


def write_embeddings(out_dir: Path, emb: np.ndarray, rows: list[dict],
                     fmt: str) -> list[str]:
    """Write embeddings as .npy (compact, for the pipeline) and/or .csv (readable)."""
    written: list[str] = []
    if fmt in ("npy", "both"):
        np.save(out_dir / "embeddings.npy", emb)
        written.append("embeddings.npy")
    if fmt in ("csv", "both"):
        kept = sorted((r for r in rows if r["status"] == "kept"),
                      key=lambda r: int(r["embedding_row"]))
        header = ["actor_id", "image_id"] + [f"e{i:03d}" for i in range(emb.shape[1])]
        with open(out_dir / "embeddings.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r, vec in zip(kept, emb):
                w.writerow([r["actor_id"], r["image_id"]]
                           + [f"{v:.6f}" for v in vec])
        written.append("embeddings.csv")
    return written


def main() -> int:
    here = Path(__file__).resolve()
    root = here.parent.parent                        # script sits in FaceModel/
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(root / "pictures"))
    ap.add_argument("--out-dir", default=str(root / "dataset"))
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--min-face-px", type=int, default=80)
    ap.add_argument("--ctx-id", type=int, default=0, help="-1 for CPU")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--emb-format", choices=("npy", "csv", "both"), default="both",
                    help="npy = compact binary (pipeline), csv = readable "
                         "(inspect in VS Code / train directly)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.is_dir():
        print(f"no such images dir: {images_dir}", file=sys.stderr)
        return 1

    id_map = load_id_map(root)
    app = load_app(args.ctx_id, args.det_size)

    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    actor_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
    if not actor_dirs:
        print(f"no actor subfolders under {images_dir}", file=sys.stderr)
        return 1

    for actor_dir in actor_dirs:
        folder = actor_dir.name
        actor_id = resolve_actor_id(folder, id_map)
        if actor_id != folder:
            print(f"  folder {folder!r} -> actor_id {actor_id!r}", file=sys.stderr)
        elif folder.lower() not in id_map:
            print(f"  warning: folder {folder!r} matches no actor_id/initials "
                  f"in dataset CSVs; using it verbatim", file=sys.stderr)
        files = sorted(p for p in actor_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXT)
        for path in files:
            img = cv2.imread(str(path))
            row = {
                "image_id": f"{actor_id}/{path.name}",
                "actor_id": actor_id,
                "path": str(path.relative_to(root)),
                "det_score": "", "bbox": "", "n_faces": "",
                "embedding_row": "", "status": "rejected",
                "reason": "", "qc_flag": "",
            }
            if img is None:
                row["reason"] = "unreadable"
                rows.append(row)
                continue

            faces = app.get(img)
            face, reason, n_faces = pick_main_face(
                faces, args.min_score, args.min_face_px)
            row["n_faces"] = n_faces
            if face is None:
                row["reason"] = reason
                rows.append(row)
                continue

            row["det_score"] = round(float(face.det_score), 4)
            row["bbox"] = ",".join(str(int(v)) for v in face.bbox)
            row["embedding_row"] = len(embeddings)
            row["status"] = "kept"
            if n_faces > 1:
                row["qc_flag"] = "multi_face"
            embeddings.append(face.normed_embedding.astype(np.float32))
            rows.append(row)

    emb = (np.vstack(embeddings) if embeddings
           else np.zeros((0, 512), dtype=np.float32))

    # --- intra-actor consistency -------------------------------------------
    by_actor: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["status"] == "kept":
            by_actor.setdefault(r["actor_id"], []).append(i)

    for actor_id, idxs in by_actor.items():
        if len(idxs) < 2:
            continue
        E = emb[[rows[i]["embedding_row"] for i in idxs]]   # already unit norm
        sim = E @ E.T
        np.fill_diagonal(sim, np.nan)
        mean_sim = np.nanmean(sim, axis=1)
        med = float(np.median(mean_sim))
        for k, i in enumerate(idxs):
            if med - mean_sim[k] > OUTLIER_MARGIN:
                tags = [t for t in rows[i]["qc_flag"].split(";") if t]
                rows[i]["qc_flag"] = ";".join(tags + ["outlier"])

    written = write_embeddings(out_dir, emb, rows, args.emb_format)
    with open(out_dir / "images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # --- report -------------------------------------------------------------
    print(f"actors: {len(actor_dirs)}   images seen: {len(rows)}   "
          f"kept: {int((np.array([r['status'] for r in rows]) == 'kept').sum())}")
    print(f"wrote {out_dir/'images.csv'}, {', '.join(written)}  {emb.shape}\n")
    for actor_id in sorted(by_actor):
        idxs = by_actor[actor_id]
        print(f"  {actor_id:<18} kept {len(idxs)}")
        for i in idxs:
            r = rows[i]
            flag = f"  [{r['qc_flag']}]" if r["qc_flag"] else ""
            print(f"      {r['image_id']:<34} score {r['det_score']:.3f} "
                  f"faces {r['n_faces']}{flag}")
        for i, r in enumerate(rows):
            if r["actor_id"] == actor_id and r["status"] == "rejected":
                print(f"      {r['image_id']:<34} REJECTED ({r['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
