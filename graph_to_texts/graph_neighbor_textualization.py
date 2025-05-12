import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import openai
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paragraph generation using OpenAI                                           #
# --------------------------------------------------------------------------- #
PROMPT_ES = (
    "Dada una lista de triplets de un grafo de conocimiento, genera un texto "
    "que describa la información contenida en estas relaciones.\n\n"
    "Triplets:\n{triplets}\n"
    "Texto:"
)

PROMPT_EN = (
    "Given a list of knowledge graph triplets, generate a text that describes "
    "the information contained in these relationships.\n\n"
    "Triplets:\n{triplets}\n"
    "Text:"
)


def generate_paragraph_from_triplets(triplets: List[str], lang: str, model: str):
    """
    Produce one paragraph that summarises *triplets* using the fixed prompts.
    """
    prompt = PROMPT_ES.format(triplets="\n".join(triplets)) if lang == "es" else PROMPT_EN.format(
        triplets="\n".join(triplets)
    )

    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
        n=1,
        stop=None,
    )
    return completion.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
def neighbour_files(base: Path, folders: Iterable[str]):
    """Return every *.pkl file under the requested folders."""
    files: List[Path] = []
    for folder in folders:
        dir_path = base / folder
        files.extend(sorted(dir_path.glob("*.pkl")))
    return files


def graph_to_paragraphs(neighbourhoods: List[Tuple[str, List[str]]], lang: str, model: str):
    """
    Convert every neighbourhood to a paragraph.
    Returns a list of ``(entity, paragraph)`` tuples.
    """
    results: List[Tuple[str, str]] = []
    for entity, triplets in tqdm(neighbourhoods, desc="Neighbours"):
        paragraph = generate_paragraph_from_triplets(triplets, lang, model)
        results.append((entity, paragraph))
    return results


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Neighbour-level KG-to-text converter")
    parser.add_argument("--case", required=True, help="Name of the collection (lonquen | 20_docs | san_gregorio)")
    parser.add_argument("--folders", nargs="+", required=True, help="Folders inside data/kgn_triplets/<case>/ that contain neighbour *.pkl files")
    parser.add_argument("--lang", choices=["es", "en"], default="es", help="Language for the generated paragraphs")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat-completion model to use")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment.")

    base_neigh_dir = Path("data/kgn_triplets") / args.case
    out_dir = Path("data/textualization_neighbors")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in neighbour_files(base_neigh_dir, args.folders):
        with pkl_path.open("rb") as fh:
            neighbourhoods = pickle.load(fh)  # List[(entity, triplet_list)]

        # Convert and save
        paragraphs = graph_to_paragraphs(neighbourhoods, lang=args.lang, model=args.model)
        outfile = out_dir / pkl_path.name
        with outfile.open("wb") as fh:
            pickle.dump(paragraphs, fh)

        print(f"Saved {len(paragraphs):,} paragraphs → {outfile}")

    print("Done!")
