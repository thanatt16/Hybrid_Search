import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# koren projektu, at jdou importy a cesty k datum
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bm25 import build_index, search_tokens
from embeddings import load_model, load_embeddings, embed_queries, dense_search
from hybrid import CANDIDATES, reciprocal_rank_fusion, W_DENSE
# stejne dotazy a metriky jako v evaluate_bm25 at se to da porovnat
from evaluate_bm25 import CONFIGS, K, preprocess_variant, load_known_item_queries, known_item_metrics

# chunker/embeddings.py importuje chunking.py ze sve slozky, proto ji pridavam do cesty
# import pres "chunker." protoze embeddings.py je i v rootu
sys.path.append(str(ROOT / "chunker"))
from chunker.embeddings import load_embeddings as load_chunk_embeddings, dense_search as dense_search_chunks

# porovnani bm25, hybridu nad celymi clanky a hybridu nad chunky

RESULTS_DIR = ROOT / "eval_results"
# varianta bm25 co jde do hybridu
BEST_CONFIG = "5 + stemming (full)"


def bm25_rankings(texts, params, queries):
    """poradi clanku z bm25 pro kazdy dotaz, spocita se jednou"""
    index = build_index([preprocess_variant(t, **params) for t in texts])
    return {q: [i for i, _ in search_tokens(preprocess_variant(q, **params), index, top_n=CANDIDATES)] for q in queries}


def evaluate_system(search, query_sets):
    """search(query) -> serazene doc_id
    vraci metriky a mrr@10 pro kazdy dotaz (pro parove srovnani)"""
    result, per_query_mrr = {}, {}

    for name, queries in query_sets.items():
        per_query = pd.DataFrame([known_item_metrics(search(q["query"])[:K], q["doc_id"]) for q in queries])
        per_query_mrr[name] = per_query["MRR@10"].to_numpy()
        for metric, value in per_query.mean().items():
            result[(name, metric)] = value

        # mrr i podle typu dotazu (synonym, typo, ...)
        if "type" in queries[0]:
            per_query["type"] = [q["type"] for q in queries]
            for qtype, value in per_query.groupby("type")["MRR@10"].mean().items():
                result[(f"{name} MRR@10 podle typu", qtype)] = value

    return result, per_query_mrr


def paired_comparison(base, new, n_boot=2000):
    """rozdil mrr po dotazech (new - base) + 95% bootstrap interval
    kdyz je v intervalu 0 tak to muze byt nahoda"""
    diff = new - base
    boot = np.random.default_rng(0).choice(diff, size=(n_boot, len(diff))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "ΔMRR@10": round(float(diff.mean()), 3),
        "95% CI": f"[{low:+.3f}, {high:+.3f}]",
        "lepší": int((diff > 0).sum()),
        "horší": int((diff < 0).sum()),
        "stejně": int((diff == 0).sum()),
    }


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "dataset" / "articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    # doc_id musi sedet s poradim v df i v embeddings.npy
    assert (df["doc_id"].to_numpy() == range(len(df))).all(), "doc_id neodpovídá pozici článku"

    query_sets = load_known_item_queries()
    all_queries = list({q["query"] for qs in query_sets.values() for q in qs})

    model = load_model()
    embeddings = load_embeddings(model, df)
    assert len(embeddings) == len(df), "embeddings.npy neodpovídá datasetu, smaž ho a spočítej znovu"

    # chunky - stejny model, jen 1 vektor na chunk misto na clanek
    chunk_embeddings, chunk_doc_ids = load_chunk_embeddings(model, df)
    assert chunk_doc_ids.max() == len(df) - 1, "chunk_doc_ids neodpovídá datasetu, smaž embeddings_chunks.npy a chunk_doc_ids.npy"

    # vsechny dotazy najednou, je to rychlejsi
    query_vectors = embed_queries(model, all_queries)
    dense = {q: dense_search(embeddings, v, top_k=CANDIDATES) for q, v in zip(all_queries, query_vectors)}
    # stejne vektory dotazu, clanek dostane poradi podle nejlepsiho chunku
    dense_chunks = {q: dense_search_chunks(chunk_embeddings, chunk_doc_ids, v, top_k=CANDIDATES)
                    for q, v in zip(all_queries, query_vectors)}

    start = time.time()
    best = bm25_rankings(texts, CONFIGS[BEST_CONFIG], all_queries)
    print(f"BM25 {BEST_CONFIG} index + dotazy za {time.time() - start:.1f} s")

    # bm25, hybrid nad clanky a hybrid nad chunky (bm25 je porad nad celymi clanky)
    # vaha embeddingu 0.7, bm25 0.3
    best_name = f"BM25 {BEST_CONFIG}"
    hybrid_name = f"hybrid RRF dense={W_DENSE}"
    chunky_name = f"hybrid chunky dense={W_DENSE}"
    systems = {
        best_name: lambda q: best[q],
        hybrid_name: lambda q: [
            d for d, _ in reciprocal_rank_fusion(best[q], dense[q], w_bm25=1 - W_DENSE, w_dense=W_DENSE)],
        chunky_name: lambda q: [
            d for d, _ in reciprocal_rank_fusion(best[q], dense_chunks[q], w_bm25=1 - W_DENSE, w_dense=W_DENSE)],
    }

    rows, per_query = {}, {}
    for name, search in systems.items():
        rows[name], per_query[name] = evaluate_system(search, query_sets)

    results = pd.DataFrame(rows).T
    results.columns = pd.MultiIndex.from_tuples(results.columns)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print()
    for group in results.columns.get_level_values(0).unique():
        print(f"=== {group} ===")
        print(results[group].round(3).to_string())
        print()

    # (zaklad, novy): hybridy proti bm25 a chunky proti celym clankum
    comparisons = [(best_name, hybrid_name), (best_name, chunky_name), (hybrid_name, chunky_name)]
    compare = pd.DataFrame([
        {"srovnání": f"{new} vs {base}", "sada": name, **paired_comparison(per_query[base][name], per_query[new][name])}
        for base, new in comparisons for name in query_sets
    ])
    print("=== párové srovnání (MRR@10 po dotazech) ===")
    print(compare.to_string(index=False))

    # mrr@10 pro kazdy dotaz, at je videt co kde selhava
    per_query_table = pd.DataFrame([
        {"sada": name, "query": q["query"], "doc_id": q["doc_id"], "type": q.get("type", ""),
         **{system: per_query[system][name][i] for system in systems}}
        for name, queries in query_sets.items() for i, q in enumerate(queries)
    ])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "results.csv"))
    compare.to_csv(os.path.join(RESULTS_DIR, "paired_comparison.csv"), index=False)
    per_query_table.to_csv(os.path.join(RESULTS_DIR, "per_query.csv"), index=False)
    print(f"\nVýsledky uloženy do {RESULTS_DIR}")
