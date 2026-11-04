import pandas as pd

from preprocessing import preprocess
from bm25 import build_index, search_tokens
from embeddings import load_model, load_embeddings, embed_queries, dense_search

# kolik kandidátů jde z BM25 a z embeddingů do RRF
CANDIDATES = 100

def reciprocal_rank_fusion(bm25_results, dense_results, k=60, w_bm25=1.0, w_dense=1.0):
    scores = {}                                   # slovník {doc_id: body}, zatím prázdný
    for rank, doc in enumerate(bm25_results, start=1):     # projde BM25 seznam, rank = 1, 2, 3, ...
        # scores.get(doc, 0.0) = dosavadní body článku, 0 pokud tam ještě není
        scores[doc] = scores.get(doc, 0.0) + w_bm25 / (k + rank)

    for rank, doc in enumerate(dense_results, start=1):    # projde seznam z embeddingů
        # pokud článek už má body z BM25, přičtou se k nim, jinak se založí nový záznam
        scores[doc] = scores.get(doc, 0.0) + w_dense / (k + rank)

    # slovník -> list dvojic (doc_id, body), např. [(835, 0.0163), (0, 0.0161), ...]
    vysledky = list(scores.items())
    # seřadí podle bodů (x[1]) od největšího, stejně jako search_tokens v bm25.py
    vysledky.sort(key=lambda x: x[1], reverse=True)
    return vysledky                               # list (doc_id, skóre), 100 až 200 položek

# váha embeddingů v RRF (BM25 dostane 1 - 0.7 = 0.3), vybráno podle evaluate_hybrid.py
W_DENSE = 0.7

def search(query, index, model, embeddings, top_n=5):

    query_tokens = preprocess(query)              # "banky v česku" -> ['bank', 'cesk']
    bm25_answer = search_tokens(query_tokens, index, top_n=CANDIDATES)   # max 100 dvojic (doc_id, BM25 skóre), jen články se skóre > 0

    bm25_results = []
    for doc_id, score_value in bm25_answer:       # rozbalí každou dvojici na doc_id a skóre
        bm25_results.append(doc_id)               # uloží jen doc_id -> [835, 0, 242, ...]

    # 2) embeddingy: dotaz převedeme na vektor a najdeme nejpodobnější články
    query_vectors = embed_queries(model, [query]) # model bere list dotazů -> matice (1, 1024)
    query_vector = query_vectors[0]               # jediný řádek -> vektor (1024,), aby šel násobit s (1000, 1024)
    dense_results = dense_search(embeddings, query_vector, top_k=CANDIDATES)   # vždy přesně 100 doc_id podle kosinové podobnosti

    # 3) spojení obou seznamů: BM25 s vahou 0.3, embeddingy s vahou 0.7
    vysledky = reciprocal_rank_fusion(bm25_results, dense_results, w_bm25=1 - W_DENSE, w_dense=W_DENSE)
    return vysledky[:top_n]                       # jen top_n (5) nejlepších dvojic (doc_id, skóre)

if __name__ == "__main__":
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    index = build_index([preprocess(text) for text in texts])
    model = load_model()
    embeddings = load_embeddings(model, df)

    while True:
        query = input("\nHledat: ")
        answer = search(query, index, model, embeddings, top_n=5)
        if not answer:
            print("Nic nenalezeno.")
        for poradi, (i, score_value) in enumerate(answer, start=1):
            print(f"{poradi}. [{score_value:.4f}] {df['headline'].iloc[i]}")
            print(f"Odkaz zde -> {df['url'].iloc[i]}")
