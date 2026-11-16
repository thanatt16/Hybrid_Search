import pandas as pd

from preprocessing import preprocess
from bm25 import build_index, search_tokens
from embeddings import load_model, load_embeddings, embed_queries, dense_search

# kolik kandidatu beru z bm25 a z embeddingu do rrf
CANDIDATES = 100

def reciprocal_rank_fusion(bm25_results, dense_results, k=60, w_bm25=1.0, w_dense=1.0):
    scores = {}                                   # {doc_id: body}
    for rank, doc in enumerate(bm25_results, start=1):     # projde BM25 seznam, rank = 1, 2, 3, ...
        # kdyz tam clanek jeste neni tak 0
        scores[doc] = scores.get(doc, 0.0) + w_bm25 / (k + rank)

    for rank, doc in enumerate(dense_results, start=1):    # ted embeddingy
        # kdyz uz ma body z bm25 tak se prictou
        scores[doc] = scores.get(doc, 0.0) + w_dense / (k + rank)

    # dict -> list dvojic (doc_id, body)
    vysledky = list(scores.items())
    # seradim podle bodu od nejvetsiho
    vysledky.sort(key=lambda x: x[1], reverse=True)
    return vysledky                               # 100 az 200 clanku

# vaha embeddingu v rrf, bm25 ma 0.3 (vybrano podle evaluace)
W_DENSE = 0.7

def search(query, index, model, embeddings, top_n=5):

    query_tokens = preprocess(query)              # "banky v cesku" -> ['bank', 'cesk']
    bm25_answer = search_tokens(query_tokens, index, top_n=CANDIDATES)   # max 100 clanku se skore > 0

    bm25_results = []
    for doc_id, score_value in bm25_answer:       # chci jen doc_id
        bm25_results.append(doc_id)               # [835, 0, 242, ...]

    # 2) embeddingy - dotaz na vektor a najdu nejpodobnejsi clanky
    query_vectors = embed_queries(model, [query]) # model chce list -> (1, 1024)
    query_vector = query_vectors[0]               # (1024,) aby to slo vynasobit s (1000, 1024)
    dense_results = dense_search(embeddings, query_vector, top_k=CANDIDATES)   # 100 doc_id podle kosinove podobnosti

    # 3) spojim to pres rrf, bm25 0.3 a embeddingy 0.7
    vysledky = reciprocal_rank_fusion(bm25_results, dense_results, w_bm25=1 - W_DENSE, w_dense=W_DENSE)
    return vysledky[:top_n]                       # top 5

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
