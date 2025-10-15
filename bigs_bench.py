import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int, overlap: int):
    """
    Split *text* into word-level chunks of length *size*
    with *overlap* words shared between consecutive chunks.
    """
    tokens = text.split()
    step = size - overlap
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens), step)]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_category_names(bench: str, split_type: str):
    """Return all category IDs available for the given benchmark and split."""
    base = Path("data/text2kgbench")
    folder = base / bench / split_type / "sentences"
    return [p.stem.split("_")[-1] for p in folder.iterdir() if p.is_file()]


def read_files(bench: str, split_type: str, category: str, split: str, *, chunk_size: int = 15, overlap: int = 2):
    """Load and optionally split the originals; return originals and generated texts."""
    base_sent = Path("data/text2kgbench") / bench / split_type / "sentences"
    base_kg = Path("data/textualization") / bench / split_type

    originals_path = base_sent / f"sentences_{bench}_{split_type}_{category}.txt"
    kg_path = base_kg / f"triples_{bench}_{split_type}_{category}.pkl"

    with originals_path.open(encoding="utf-8") as fh:
        originals = [line.strip() for line in fh]

    with kg_path.open("rb") as fh:
        generated = pickle.load(fh)

    # Optional chunking
    if split == "chunks":
        originals = chunk_text(" ".join(originals), chunk_size, overlap)

    return originals, generated


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def bigs_scores(originals: List[str], generated: List[str], model: SentenceTransformer, batch_size: int):
    """Compute BIGS left/right scores and basic statistics."""
    start = time.time()
    orig_emb = model.encode(originals, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    gen_emb = model.encode(generated, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    elapsed = time.time() - start

    distances = cdist(orig_emb.cpu(), gen_emb.cpu(), metric="cosine")

    # Right (document → graph)
    right_min = distances.min(axis=1)
    score_r = right_min.mean()
    score_r_std = right_min.std()
    score_r_med = float(np.median(right_min))

    # Left (graph → document)
    left_min = distances.min(axis=0)
    score_l = left_min.mean()
    score_l_std = left_min.std()
    score_l_med = float(np.median(left_min))

    print(f"BIGS→  mean={score_r:.4f}  std={score_r_std:.4f}  median={score_r_med:.4f}")
    print(f"BIGS←  mean={score_l:.4f}  std={score_l_std:.4f}  median={score_l_med:.4f}")
    return (score_r, score_r_std, score_r_med,
        score_l, score_l_std, score_l_med, elapsed)


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    nltk.download("punkt", quiet=True)

    parser = argparse.ArgumentParser("BIGS scorer for Text2KGBench")
    parser.add_argument("--model_name", default="all-mpnet-base-v2", help="Sentence-BERT model.")
    parser.add_argument("--bench_name", default="webwiki", choices=["webwiki", "webnlg"])
    parser.add_argument("--split_type", default="test", help="Dataset split (e.g. train/valid/test).")
    parser.add_argument("--split", default="sentences", choices=["sentences", "chunks"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_name", default="results_text2kg", help="CSV prefix for results.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)

    categories = load_category_names(args.bench_name, args.split_type)
    output_path = Path(f"{args.output_name}_{args.bench_name}_{args.split_type}_{args.split}.csv")

    header = ["category", "n_original", "n_generated", "split", "model_name", "emb_dim", "batch_size",
        "bigs_left", "bigs_left_std", "bigs_left_median", "bigs_right", "bigs_right_std", "bigs_right_median", "exec_time_s"]
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)

        for cat in categories:
            print(f"\nCategory: {cat}")
            originals, generated = read_files(
                args.bench_name, args.split_type, cat, args.split
            )
            print(f"# original units:  {len(originals):,}")
            print(f"# generated texts: {len(generated):,}")

            (right_mean, right_std, right_med, left_mean, left_std, left_med, elapsed) = bigs_scores(originals, generated, model, args.batch_size)

            writer.writerow(
                [
                    cat, len(originals), len(generated), args.split, args.model_name,
                    model.get_sentence_embedding_dimension(), args.batch_size,
                    round(left_mean, 4), round(left_std, 4), round(left_med, 4),
                    round(right_mean, 4), round(right_std, 4), round(right_med, 4), round(elapsed, 2)
                ]
            )
            print(f"✓  Saved results for '{cat}'")

    print(f"\nResults appended to  {output_path.resolve()}")


