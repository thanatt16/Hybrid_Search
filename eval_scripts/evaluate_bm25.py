import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

# koren projektu, at jdou importy bm25/preprocessing a cesty k datum
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bm25 import build_index, search_tokens
from preprocessing import (preprocess, remove_punctuation, tokenize, lemmatize, lowercase, strip_diacritics, remove_stopwords, stem)

K = 10
# varianty preprocessingu, 0 = jen interpunkce + tokenizace, kazda dalsi prida jeden krok
CONFIGS = {
    "0 baseline":                 dict(lower=False, lemma=False, fold_diacritics=False, stopwords=False, stemming=False),
    "1 + malá písmena":           dict(lower=True,  lemma=False, fold_diacritics=False, stopwords=False, stemming=False),
    "2 + bez diakritiky":         dict(lower=True,  lemma=False, fold_diacritics=True,  stopwords=False, stemming=False),
    "3 + stopwords":              dict(lower=True,  lemma=False, fold_diacritics=True,  stopwords=True,  stemming=False),
    "4 + lemmatizace":            dict(lower=True,  lemma=True,  fold_diacritics=True,  stopwords=True,  stemming=False),
    "5 + stemming (full)":        dict(lower=True,  lemma=True,  fold_diacritics=True,  stopwords=True,  stemming=True),
    "6 stemming bez lemmatizace": dict(lower=True,  lemma=False, fold_diacritics=True,  stopwords=True,  stemming=True),
}


def preprocess_variant(text, lower=True, lemma=True, fold_diacritics=True, stopwords=True, stemming=True):
    """to same co preprocess() ale jde vypnout kazdy krok"""
    if lower and fold_diacritics and stopwords:
        return preprocess(text, lemma=lemma, stemming=stemming)

    tokens = tokenize(remove_punctuation(text))
    if lemma:
        tokens = lemmatize(tokens)
        tokens = [t for t in tokens if t != "být"]
    if lower:
        tokens = lowercase(tokens)
    if fold_diacritics:
        tokens = [strip_diacritics(t) for t in tokens]
    if stopwords:
        tokens = remove_stopwords(tokens)
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def load_known_item_queries() -> dict[str, list[dict]]:
    with open(ROOT / "eval_data" / "queries_known_item.json", encoding="utf-8") as f:
        manual = json.load(f)

    # stejne dotazy bez diakritiky
    manual_ascii = [{"query": strip_diacritics(q["query"]), "doc_id": q["doc_id"]} for q in manual]

    # dotazy z keywords clanku
    with open(ROOT / "eval_data" / "queries_keywords.json", encoding="utf-8") as f:
        keywords = json.load(f)

    # 500 tezkych dotazu, malo spolecnych slov s titulkem a perexem
    with open(ROOT / "eval_data" / "queries_hard.json", encoding="utf-8") as f:
        hard = json.load(f)

    # 120 dotazu na stred a konec dlouhych clanku (1500+ tokenu)
    # middle / end = se jmeny z pasaze, *_paraphrase = stejna pasaz jinymi slovy
    with open(ROOT / "eval_data" / "queries_position.json", encoding="utf-8") as f:
        position = json.load(f)

    return {"ruční(přirozené)": manual, "ruční bez diakritiky": manual_ascii, "keywords (auto)": keywords,
            "těžké (500)": hard, "střed a konec (120)": position}


def known_item_metrics(ranking: list[int], relevant: int) -> dict[str, float]:
    """ranking = serazene doc_id, relevant = spravny doc_id"""
    rank = ranking.index(relevant) + 1 if relevant in ranking[:K] else None
    return {
        "MRR@10": 1 / rank if rank else 0.0,
        "R@1": float(rank == 1),
        "R@5": float(rank is not None and rank <= 5),
        "R@10": float(rank is not None),
        # jen 1 spravny clanek takze idealni dcg = 1
        "nDCG@10": 1 / math.log2(rank + 1) if rank else 0.0,
    }


def rank_docs(index: dict, query_tokens: list[str]) -> list[int]:
    return [i for i, _ in search_tokens(query_tokens, index, top_n=K)]


def evaluate_config(texts: pd.Series, params: dict, query_sets: dict) -> dict:
    corpus = [preprocess_variant(t, **params) for t in texts]
    index = build_index(corpus)

    result = {}
    for name, queries in query_sets.items():
        per_query = pd.DataFrame(
            [known_item_metrics(rank_docs(index, preprocess_variant(q["query"], **params)), q["doc_id"]) for q in queries]
        )
        for metric, value in per_query.mean().items():
            result[(name, metric)] = value

        # mrr i podle typu dotazu (synonym, typo, ...)
        if "type" in queries[0]:
            per_query["type"] = [q["type"] for q in queries]
            for qtype, value in per_query.groupby("type")["MRR@10"].mean().items():
                result[(f"{name} MRR@10 podle typu", qtype)] = value

    return result


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "dataset" / "articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    # doc_id musi sedet s poradim v df
    assert (df["doc_id"].to_numpy() == range(len(df))).all(), "doc_id neodpovídá pozici článku"

    query_sets = load_known_item_queries()

    rows = {}
    for name, params in CONFIGS.items():
        start = time.time()
        rows[name] = evaluate_config(texts, params, query_sets)
        print(f"{name:28} hotovo za {time.time() - start:.1f} s")

    results = pd.DataFrame(rows).T
    results.columns = pd.MultiIndex.from_tuples(results.columns)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print()
    for group in results.columns.get_level_values(0).unique():
        print(f"=== {group} ===")
        print(results[group].round(3).to_string())
        print()

    results.to_csv(ROOT / "eval_results" / "results_bm25.csv")
