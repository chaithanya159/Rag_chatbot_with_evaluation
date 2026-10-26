"""
experiment.py
-------------
Retrieval-only experiments. For several chunking settings it rebuilds the
index IN MEMORY (your saved index and the running app are not touched),
runs every question in the chosen test set (see evaluate.py), and reports how well the right
pages / passages come back.

No LLM calls -> no API key needed, no rate limits, finishes in a minute or two.

Usage (from the rag_flask folder, venv active):
    python experiment.py                          # uses the only PDF in this folder
    python experiment.py "your_file.pdf"          # or give its name / full path
    python experiment.py --set holdout            # score the hold-out questions instead

Metrics (all computed from the ranked chunks, no judge involved):
    hit@k     share of answerable questions where at least one of the top-k
              chunks comes from an expected page
    MRR       mean reciprocal rank of the first chunk from an expected page
              (1.0 = always rank 1, 0.5 = typically rank 2, 0 = never in top-MAX_K)
    kw@5      average keyword coverage of the top-5 chunks (same idea as evaluate.py)
    probe     rank of the first chunk containing a specific phrase (the Q8 needle)
"""

import os
import sys
import csv

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import loaders.document_loader as dl
from rag.embeddings import EmbeddingManager
from evaluate import TEST_SETS

# ---------------------------------------------------------------------------
# Settings you can edit
# ---------------------------------------------------------------------------
CHUNK_SIZES = [100, 150, 200, 300]   # words per chunk  (300 = current app setting)
OVERLAPS = [25, 50]                  # words repeated between chunks (50 = current)
K_VALUES = [3, 5, 8]                 # top-k values to report
BASELINE = (300, 50)                 # (size, overlap) currently used by the app

# "Needles": a phrase that must appear in a retrieved chunk for the question to
# really be answerable. Reports the rank of the first chunk containing it.
PROBES = [
    {"id": "Q8", "phrase": "establishes the context"},   # main set
    {"id": "H8", "phrase": "establishes the context"},   # hold-out set (same passage)
]

def results_csv(set_name):
    name = "experiment_results.csv" if set_name == "main" else f"experiment_results_{set_name}.csv"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
MAX_K = max(K_VALUES + [10])


def build_chunks(pdf_path: str, size: int, overlap: int):
    """Chunk the PDF with the app's own chunker using the given settings."""
    old_size, old_overlap = dl.CHUNK_SIZE, dl.CHUNK_OVERLAP
    dl.CHUNK_SIZE, dl.CHUNK_OVERLAP = size, overlap
    try:
        return dl.load_document(pdf_path)
    finally:
        dl.CHUNK_SIZE, dl.CHUNK_OVERLAP = old_size, old_overlap


def evaluate_config(embedder, chunks, cases, probes):
    texts = [c.page_content for c in chunks]
    chunk_vecs = embedder.generate_embeddings(texts)
    index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    index.add(chunk_vecs)

    questions = [case["question"] for case in cases]
    query_vecs = embedder.generate_embeddings(questions)
    k = min(MAX_K, len(chunks))
    _, ids = index.search(query_vecs, k)

    hits = {kk: [] for kk in K_VALUES}
    rr_list, kw_list = [], []
    misses_at_5 = []
    probe_ranks = {p["id"]: None for p in probes}

    for row, case in enumerate(cases):
        ranked = [chunks[i] for i in ids[row]]
        expected_pages = set(case.get("expected_pages", []))

        if expected_pages:
            pages_ranked = [c.metadata.get("page") for c in ranked]
            for kk in K_VALUES:
                hits[kk].append(1.0 if expected_pages & set(pages_ranked[:kk]) else 0.0)
            first = next((r for r, p in enumerate(pages_ranked, start=1) if p in expected_pages), None)
            rr_list.append(1.0 / first if first else 0.0)
            if not (expected_pages & set(pages_ranked[:5])):
                misses_at_5.append(case["id"])

        keywords = case.get("expected_keywords", [])
        if keywords:
            top5_text = " ".join(c.page_content for c in ranked[:5]).lower()
            found = [kw for kw in keywords if kw.lower() in top5_text]
            kw_list.append(len(found) / len(keywords))

        for probe in probes:
            if probe["id"] == case.get("id"):
                phrase = probe["phrase"].lower()
                rank = next((r for r, c in enumerate(ranked, start=1)
                             if phrase in c.page_content.lower()), None)
                probe_ranks[probe["id"]] = rank

    def mean(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    return {
        "chunks": len(chunks),
        **{f"hit@{kk}": mean(hits[kk]) for kk in K_VALUES},
        "MRR": mean(rr_list),
        "kw@5": mean(kw_list),
        "probes": probe_ranks,
        "misses@5": misses_at_5,
    }


def fmt_rank(r):
    return f">{MAX_K}" if r is None else str(r)


def main(embedder=None, pdf_path=None, set_name=None):
    if pdf_path is None and set_name is None:
        import argparse
        parser = argparse.ArgumentParser(description="Retrieval-only chunking experiments.")
        parser.add_argument("pdf", nargs="?", help="PDF to index (default: the only PDF in this folder)")
        parser.add_argument("--set", choices=sorted(TEST_SETS), default="main",
                            help="which test set to score (default: main)")
        args = parser.parse_args()
        pdf_path, set_name = args.pdf, args.set
    set_name = set_name or "main"
    cases = TEST_SETS[set_name]
    probes = [p for p in PROBES if p["id"] in {c.get("id") for c in cases}]

    if pdf_path is None:
        # No path given: use the only PDF sitting next to this script.
        here = os.path.dirname(os.path.abspath(__file__))
        found = sorted(f for f in os.listdir(here) if f.lower().endswith(".pdf"))
        if len(found) == 1:
            pdf_path = os.path.join(here, found[0])
        elif not found:
            print("No PDF found in this folder. Copy the NIST PDF here, or pass its path.")
            return []
        else:
            print("More than one PDF in this folder; pass the one to use:")
            for f in found:
                print(f'  python experiment.py "{f}"')
            return []
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        print("Check the file name/path (use the real path, not the example one).")
        return []

    print("=" * 96)
    print(f"RETRIEVAL EXPERIMENT (no LLM calls) - test set: {set_name}, {len(cases)} questions")
    if set_name == "holdout":
        print("HOLD-OUT: use this to CHECK a setting already chosen on the main set, not to pick one.")
    print(f"PDF: {pdf_path}")
    print("=" * 96)

    embedder = embedder or EmbeddingManager()
    probe_labels = [f"{p['id']} rank" for p in probes]

    header = f"{'config':<20}{'chunks':>7}" + "".join(f"{'hit@' + str(k):>8}" for k in K_VALUES) \
             + f"{'MRR':>8}{'kw@5':>8}" + "".join(f"{lbl:>10}" for lbl in probe_labels)
    print(header)
    print("-" * len(header))

    rows = []
    for size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            if overlap >= size:
                continue
            chunks = build_chunks(pdf_path, size, overlap)
            m = evaluate_config(embedder, chunks, cases, probes)
            label = f"size={size} ov={overlap}" + ("  <- now" if (size, overlap) == BASELINE else "")
            line = f"{label:<20}{m['chunks']:>7}" + "".join(f"{m['hit@' + str(k)]:>8}" for k in K_VALUES) \
                   + f"{m['MRR']:>8}{m['kw@5']:>8}" \
                   + "".join(f"{fmt_rank(m['probes'][p['id']]):>10}" for p in probes)
            print(line)
            rows.append({"size": size, "overlap": overlap, **{k: v for k, v in m.items() if k != "probes"},
                         **{f"{pid}_rank": r for pid, r in m["probes"].items()}})
            m["size"], m["overlap"] = size, overlap
            rows[-1]["_full"] = m

    if not rows:
        return rows

    best = max(rows, key=lambda r: ((r["hit@5"] or 0), (r["MRR"] or 0), -r["chunks"]))
    base = next((r for r in rows if (r["size"], r["overlap"]) == BASELINE), None)

    print("\nBest by hit@5 (ties: MRR, then fewer chunks): "
          f"size={best['size']} overlap={best['overlap']}  "
          f"(hit@5={best['hit@5']}, MRR={best['MRR']})")
    if base:
        print(f"Questions still missing at k=5 -> baseline: {base['misses@5'] or 'none'} | "
              f"best: {best['misses@5'] or 'none'}")

    # CSV (drop the internal helper key)
    clean = [{k: v for k, v in r.items() if k != "_full"} for r in rows]
    for r in clean:
        r["misses@5"] = " ".join(r["misses@5"])
    with open(results_csv(set_name), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
        writer.writeheader()
        writer.writerows(clean)
    print(f"\nSaved to {results_csv(set_name)}")
    print("Notes:")
    print(" - Pick settings using hit@k, MRR AND the probe rank together. With only 13 answerable")
    print("   questions, one question = 0.077, so don't over-read small differences.")
    print(" - kw@5 is NOT comparable across chunk sizes: smaller chunks put less text in the top 5,")
    print("   so coverage drops even when retrieval is fine. Use it within one chunk size only.")
    return rows


if __name__ == "__main__":
    main()
