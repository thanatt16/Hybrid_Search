import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from chunking import chunk_articles

MODEL_NAME = "BAAI/bge-m3"
EMBEDDINGS_PATH = r"E:\Machine_learning\NLP\Project\Hybrid_search\artifacts\embeddings_chunks.npy"
CHUNK_DOC_IDS_PATH = r"E:\Machine_learning\NLP\Project\Hybrid_search\artifacts\chunk_doc_ids.npy"

def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Načítám {MODEL_NAME} na {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 1024 # chunk ma max 512 tokenu + titulek+ brief, takze se vejde a nic se neorizne
    return model

def embed_chunks(model, chunks):
    return model.encode(chunks, batch_size=16, normalize_embeddings=True, show_progress_bar=True) # matice (pocet chunku, 1024), kazdy chunk = 1 vektor

def load_embeddings(model, df):
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CHUNK_DOC_IDS_PATH): # pokud jsou ulozene, nacte je z disku pro rychlost
        return np.load(EMBEDDINGS_PATH), np.load(CHUNK_DOC_IDS_PATH)
    chunks, chunk_doc_ids = chunk_articles(df) # rozdeli clanky na chunky, ke kazdemu chunku doc_id clanku
    print(f"{len(df)} článků -> {len(chunks)} chunků")
    embeddings = embed_chunks(model, chunks)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True) # vytvorim artifacts
    np.save(EMBEDDINGS_PATH, embeddings)    # matice (pocet chunku, 1024)
    np.save(CHUNK_DOC_IDS_PATH, chunk_doc_ids) # (pocet chunku,) -> doc_id clanku
    return embeddings, chunk_doc_ids

def embed_queries(model, queries):
    # stejny model i normalizace jako u chunku, jinak by to neslo porovnat
    return model.encode(queries, batch_size=32, normalize_embeddings=True) # 1 dotaz = (1, 1024)

def dense_search(embeddings, chunk_doc_ids, query_vector, top_k=100):
    # kosinova podobnost se vsema chunkama najednou, vektory jsou normalizovane takze staci skalarni soucin
    scores = embeddings @ query_vector # (pocet chunku x 1024) x (1024,) -> jedno cislo na chunk
    best = np.argsort(-scores)         # chunky od nejpodobnejsiho

    # clanek dostane poradi podle sveho nejlepsiho chunku (MaxP), kazdy clanek jen jednou
    vysledek = []
    videne = set()
    for i in best:
        doc_id = int(chunk_doc_ids[i]) # chunk -> clanek, ze ktereho je
        if doc_id in videne:           # clanek uz ma lepsi chunk vys -> preskocit
            continue
        videne.add(doc_id)
        vysledek.append(doc_id)
        if len(vysledek) == top_k:
            break
    return vysledek                    # list doc_id clanku, stejny format jako dense_search v embeddings.py
