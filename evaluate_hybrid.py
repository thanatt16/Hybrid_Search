import json
import os
import time

import numpy as np
import pandas as pd

from bm25 import build_index, search_tokens
from embeddings import load_model, load_embeddings, embed_queries, dense_search
from hybrid import CANDIDATES, reciprocal_rank_fusion, W_DENSE
# stejné varianty preprocessingu, dotazy a metriky jako evaluate_bm25 -> výsledky BM25 jsou přímo srovnatelné
from evaluate_bm25 import CONFIGS, K, preprocess_variant, load_known_item_queries, known_item_metrics

# Porovnání BM25 (lexikální) s různým preprocessingem, embeddingů (sémantické) a hybridu (RRF) na stejných dotazech.

RESULTS_DIR = r"E:\Machine_learning\NLP\Project\Hybrid_search\results"
# varianta BM25, která jde do hybridu a do párového srovnání (preprocess s výchozími parametry)
BEST_CONFIG = "5 + stemming (full)"


def bm25_rankings(texts, params, queries):
    """pořadí článků z BM25 pro každý dotaz; počítá se jednou a pak se jen čte"""
    index = build_index([preprocess_variant(t, **params) for t in texts])
    return {q: [i for i, _ in search_tokens(preprocess_variant(q, **params), index, top_n=CANDIDATES)] for q in queries}


def evaluate_system(search, query_sets, category_queries, categories):
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

    # kategorie: kolik z top 10 výsledků patří do očekávané kategorie
    precisions = []
    for q in category_queries:
        top = search(q["query"])[:K]
        precisions.append(sum(categories.iloc[i] == q["category"] for i in top) / K)
    result[("kategorie", "P@10")] = float(np.mean(precisions))

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
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    # doc_id v dotazech = pozice článku v indexu i v embeddings.npy, musí sedět s pořadím v df
    assert (df["doc_id"].to_numpy() == range(len(df))).all(), "doc_id neodpovídá pozici článku"

    query_sets = load_known_item_queries()
    with open(r"E:\Machine_learning\NLP\Project\Hybrid_search\eval\queries_category.json", encoding="utf-8") as f:
        category_queries = json.load(f)
    all_queries = list({q["query"] for qs in query_sets.values() for q in qs} | {q["query"] for q in category_queries})

    model = load_model()
    embeddings = load_embeddings(model, df)
    assert len(embeddings) == len(df), "embeddings.npy neodpovídá datasetu, smaž ho a spočítej znovu"

    # všechny dotazy se zakódují najednou, je to mnohem rychlejší než po jednom
    query_vectors = embed_queries(model, all_queries)
    dense = {q: dense_search(embeddings, v, top_k=CANDIDATES) for q, v in zip(all_queries, query_vectors)}

    bm25 = {}
    for name, params in CONFIGS.items():
        start = time.time()
        bm25[name] = bm25_rankings(texts, params, all_queries)
        print(f"BM25 {name:27} index + dotazy za {time.time() - start:.1f} s")
    best = bm25[BEST_CONFIG]

    # default argumenty kvůli lambdě v cyklu, jinak by všechny varianty použily poslední index
    systems = {f"BM25 {name}": lambda q, r=rankings: r[q] for name, rankings in bm25.items()}
    systems["embeddings (bge-m3)"] = lambda q: dense[q]
    systems["hybrid (RRF)"] = lambda q: [d for d, _ in reciprocal_rank_fusion(best[q], dense[q])]
    # váhované RRF: w = váha embeddingů, 1 - w = váha BM25 (w=0.5 dává stejné pořadí jako hybrid (RRF), proto chybí)
    for w in [0.6, 0.7, 0.8, 0.9]:
        systems[f"hybrid RRF dense={w}"] = lambda q, w=w: [
            d for d, _ in reciprocal_rank_fusion(best[q], dense[q], w_bm25=1 - w, w_dense=w)]

    rows, per_query = {}, {}
    for name, search in systems.items():
        rows[name], per_query[name] = evaluate_system(search, query_sets, category_queries, df["category"])

    results = pd.DataFrame(rows).T
    results.columns = pd.MultiIndex.from_tuples(results.columns)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print()
    for group in results.columns.get_level_values(0).unique():
        print(f"=== {group} ===")
        print(results[group].round(3).to_string())
        print()

    best_name = f"BM25 {BEST_CONFIG}"
    hybrid_name = f"hybrid RRF dense={W_DENSE}"
    comparisons = [("BM25 0 baseline", best_name), (best_name, "embeddings (bge-m3)"), (best_name, "hybrid (RRF)"),
                   ("embeddings (bge-m3)", "hybrid (RRF)"),
                   (best_name, hybrid_name), ("embeddings (bge-m3)", hybrid_name)]
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
