import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# kořen projektu (o složku výš než eval_scripts/), aby šly importy modulů z kořene a relativní cesty k datům
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bm25 import build_index, search_tokens
from embeddings import load_model, load_embeddings, embed_queries, dense_search
from hybrid import CANDIDATES, reciprocal_rank_fusion, W_DENSE
# stejné varianty preprocessingu, dotazy a metriky jako evaluate_bm25 -> výsledky BM25 jsou přímo srovnatelné
from evaluate_bm25 import CONFIGS, K, preprocess_variant, load_known_item_queries, known_item_metrics

# embeddingy chunků z chunker/; chunker/embeddings.py si importuje chunking.py ze své složky, proto ji přidáme do cesty.
# Import přes "chunker." kvůli stejnému jménu embeddings.py v kořeni a v chunker/.
sys.path.append(str(ROOT / "chunker"))
from chunker.embeddings import load_embeddings as load_chunk_embeddings, dense_search as dense_search_chunks

# Porovnání BM25 s plným preprocessingem a dvou hybridů (RRF, váha embeddingů 0.7): nad celými články a nad chunky.

RESULTS_DIR = ROOT / "eval_results"
# varianta BM25, která jde do hybridu a do párového srovnání (preprocess s výchozími parametry)
BEST_CONFIG = "5 + stemming (full)"


def bm25_rankings(texts, params, queries):
    """pořadí článků z BM25 pro každý dotaz; počítá se jednou a pak se jen čte"""
    index = build_index([preprocess_variant(t, **params) for t in texts])
    return {q: [i for i, _ in search_tokens(preprocess_variant(q, **params), index, top_n=CANDIDATES)] for q in queries}


def evaluate_system(search, query_sets):
    """search(query) -> pozice článků seřazené podle relevance.
    Vrací souhrnné metriky a MRR@10 po jednotlivých dotazech (pro párové srovnání)."""
    result, per_query_mrr = {}, {}

    for name, queries in query_sets.items():
        per_query = pd.DataFrame([known_item_metrics(search(q["query"])[:K], q["doc_id"]) for q in queries])
        per_query_mrr[name] = per_query["MRR@10"].to_numpy()
        for metric, value in per_query.mean().items():
            result[(name, metric)] = value

        # u sad s typem dotazu navíc MRR podle typu (synonym, typo, ...)
        if "type" in queries[0]:
            per_query["type"] = [q["type"] for q in queries]
            for qtype, value in per_query.groupby("type")["MRR@10"].mean().items():
                result[(f"{name} MRR@10 podle typu", qtype)] = value

    return result, per_query_mrr


def paired_comparison(base, new, n_boot=2000):
    """Rozdíl MRR po dotazech (new - base) s 95% bootstrap intervalem.
    Když interval obsahuje 0, rozdíl může být jen náhoda."""
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
    # doc_id v dotazech = pozice článku v indexu i v embeddings.npy, musí sedět s pořadím v df
    assert (df["doc_id"].to_numpy() == range(len(df))).all(), "doc_id neodpovídá pozici článku"

    query_sets = load_known_item_queries()
    all_queries = list({q["query"] for qs in query_sets.values() for q in qs})

    model = load_model()
    embeddings = load_embeddings(model, df)
    assert len(embeddings) == len(df), "embeddings.npy neodpovídá datasetu, smaž ho a spočítej znovu"

    # chunky: stejný model, jen místo 1 vektoru na článek 1 vektor na chunk (artifacts/embeddings_chunks.npy)
    chunk_embeddings, chunk_doc_ids = load_chunk_embeddings(model, df)
    assert chunk_doc_ids.max() == len(df) - 1, "chunk_doc_ids neodpovídá datasetu, smaž embeddings_chunks.npy a chunk_doc_ids.npy"

    # všechny dotazy se zakódují najednou, je to mnohem rychlejší než po jednom
    query_vectors = embed_queries(model, all_queries)
    dense = {q: dense_search(embeddings, v, top_k=CANDIDATES) for q, v in zip(all_queries, query_vectors)}
    # stejné vektory dotazů, hledá se v chunkách a článek dostane pořadí svého nejlepšího chunku
    dense_chunks = {q: dense_search_chunks(chunk_embeddings, chunk_doc_ids, v, top_k=CANDIDATES)
                    for q, v in zip(all_queries, query_vectors)}

    start = time.time()
    best = bm25_rankings(texts, CONFIGS[BEST_CONFIG], all_queries)
    print(f"BM25 {BEST_CONFIG} index + dotazy za {time.time() - start:.1f} s")

    # BM25 full, hybrid nad celými články a hybrid nad chunky (BM25 zůstává nad celými články, mění se jen embeddingová část)
    # w_dense = W_DENSE (0.7) = váha embeddingů, 1 - W_DENSE = váha BM25
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

    # (základ, nový systém): hybridy proti BM25 a chunky proti celým článkům
    comparisons = [(best_name, hybrid_name), (best_name, chunky_name), (hybrid_name, chunky_name)]
    compare = pd.DataFrame([
        {"srovnání": f"{new} vs {base}", "sada": name, **paired_comparison(per_query[base][name], per_query[new][name])}
        for base, new in comparisons for name in query_sets
    ])
    print("=== párové srovnání (MRR@10 po dotazech) ===")
    print(compare.to_string(index=False))

    # MRR@10 každého dotazu u každého systému -> je vidět, které dotazy kde selhávají
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
