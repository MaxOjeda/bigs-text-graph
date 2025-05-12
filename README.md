# BIGS — Text ⇄ Graph Experiments  
---

## 1 · Quick start

```Bash
# clone repo
git clone <repo-url> bigs && cd bigs

# create environment
curl -LsSf https://astral.sh/uv/install.sh | sh        # one-off
uv venv
uv pip install -r requirements.txt

# add OpenAI key
export OPENAI_API_KEY="sk-..."

# run the full metric grid (≈ 15 min CPU)
python run_bigs_experiments.py
```
## 2 · Project Structure (1-Depth)
```Bash
.
├── data/
│   ├── docs_graphs/                # JSON KGs
│   ├── docs_texts/                 # raw documents (.txt)
│   ├── text2kgbench/               # WebNLG / WebWiki benchmarks
│   ├── textualization/             # triple-level sentences (.pkl)
│   ├── textualization_neighbors/   # neighbor-level paragraphs (.pkl)
│   └── triplet_lists/              # KG triplets as plain text
│
├── graph_to_text/
│   ├── graph_textualization.py           # one sentence per triple
│   ├── graph_neighbor_textualization.py  # one paragraph per node
│   └── benchmarks_textualization.py      # WebNLG / WebWiki triples
│
├── bigs_score.py                 # BIGS ← / → for two corpora
├── bigs_bench.py                 # BIGS on WebNLG / WebWiki splits
├── run_bigs_experiments.py       # batch-runner for our datasets
│
├── results/                      # auto-generated CSV files
│   ├── documents/
│   └── text2kgBench/
│
└── README.md
```

## 3 · Re-create experiments
```Bash
# 1) textualise graphs (triple level) — example for lonquen
python graph_textualization.py --case lonquen \
       --folders original langchain original_res

# 2) textualise neighbour level
python graph_neighbor_textualization.py --case lonquen \
       --folders neighbors/langchain neighbors/original neighbors/original_res

# 3) textualise benchmarks
python benchmarks_textualization.py --bench webwiki --split test
python benchmarks_textualization.py --bench webnlg  --split train

# 4) compute BIGS on our corpora
python run_bigs_experiments.py

# 5) compute BIGS on benchmarks
python bigs_bench.py --bench webwiki --split test
python bigs_bench.py --bench webnlg  --split train
```
