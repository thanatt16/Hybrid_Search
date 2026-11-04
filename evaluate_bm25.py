import json
import math
import time

import pandas as pd

from bm25 import build_index, search_tokens
from preprocessing import (preprocess, remove_punctuation, tokenize, lemmatize, lowercase,
                        strip_diacritics, remove_stopwords, stem)

K = 10
# varianty preprocessingu pro ablaci (0 = jen tokenizace, od 1 je lowercase vždy zapnutý)
CONFIGS = {
    "0 baseline (raw)":          dict(lower=False, lemma=False, fold_diacritics=False, stopwords=False, stemming=False),
    "1 tokens":                  dict(lemma=False, fold_diacritics=False, stopwords=False, stemming=False),
    "2 + bez diakritiky":        dict(lemma=False, fold_diacritics=True,  stopwords=False, stemming=False),
    "3 + stopwords":             dict(lemma=False, fold_diacritics=True,  stopwords=True,  stemming=False),
    "4 + lemma":                 dict(lemma=True,  fold_diacritics=True,  stopwords=True,  stemming=False),
    "5 + lemma + stem (full)":   dict(lemma=True,  fold_diacritics=True,  stopwords=True,  stemming=True),
    "6 stem bez lemma":          dict(lemma=False, fold_diacritics=True,  stopwords=True,  stemming=True),
}


def preprocess_variant(text, lower=True, lemma=True, fold_diacritics=True, stopwords=True, stemming=True):
    """stejné kroky a pořadí jako preprocess.preprocess, jen jdou jednotlivě vypnout"""
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
    with open(r"E:\Machine_learning\NLP\Project\Hybrid_search\eval\queries_known_item.json", encoding="utf-8") as f:
        manual = json.load(f)

    # stejné dotazy, jak by je napsal uživatel bez háčků a čárek
    manual_ascii = [{"query": strip_diacritics(q["query"]), "doc_id": q["doc_id"]} for q in manual]

    # automatická sada z keywords, vytvořená skriptem make_keyword_queries.py
    with open(r"E:\Machine_learning\NLP\Project\Hybrid_search\eval\queries_keywords.json", encoding="utf-8") as f:
        keywords = json.load(f)

    # 500 těžkých dotazů s malým překryvem s titulkem a perexem, jiné články než ruční sada
    with open(r"E:\Machine_learning\NLP\Project\Hybrid_search\eval\queries_hard.json", encoding="utf-8") as f:
        hard = json.load(f)

    return {"ruční(přirozené)": manual, "ruční bez diakritiky": manual_ascii, "keywords (auto)": keywords,
            "těžké (500)": hard}


def known_item_metrics(ranking: list[int], relevant: int) -> dict[str, float]:
    """ranking = doc_id seřazené podle skóre, relevant = jediný správný doc_id"""
    rank = ranking.index(relevant) + 1 if relevant in ranking[:K] else None
    return {
        "MRR@10": 1 / rank if rank else 0.0,
        "R@1": float(rank == 1),
        "R@5": float(rank is not None and rank <= 5),
        "R@10": float(rank is not None),
        # jeden relevantní dokument -> ideální DCG = 1
        "nDCG@10": 1 / math.log2(rank + 1) if rank else 0.0,
    }


def rank_docs(index: dict, query_tokens: list[str]) -> list[int]:
    return [i for i, _ in search_tokens(query_tokens, index, top_n=K)]


def evaluate_config(categories: pd.Series, texts: pd.Series, params: dict, query_sets: dict, category_queries: list) -> dict:
    corpus = [preprocess_variant(t, **params) for t in texts]
    index = build_index(corpus)

    result = {}
    for name, queries in query_sets.items():
        per_query = pd.DataFrame(
            [known_item_metrics(rank_docs(index, preprocess_variant(q["query"], **params)), q["doc_id"]) for q in queries]
        )
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
        ranking = rank_docs(index, preprocess_variant(q["query"], **params))
        precisions.append(sum(categories.iloc[i] == q["category"] for i in ranking) / K)
    result[("kategorie", "P@10")] = sum(precisions) / len(precisions)
    return result


if __name__ == "__main__":
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    # doc_id v dotazech = pozice článku v indexu, musí sedět s pořadím v df
    assert (df["doc_id"].to_numpy() == range(len(df))).all(), "doc_id neodpovídá pozici článku"

    query_sets = load_known_item_queries()
    with open(r"E:\Machine_learning\NLP\Project\Hybrid_search\eval\queries_category.json", encoding="utf-8") as f:
        category_queries = json.load(f)

    rows = {}
    for name, params in CONFIGS.items():
        start = time.time()
        rows[name] = evaluate_config(df["category"], texts, params, query_sets, category_queries)
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

    results.to_csv(r"E:\Machine_learning\NLP\Project\Hybrid_search\results\results_bm25.csv")
