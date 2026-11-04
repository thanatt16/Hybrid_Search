import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EMBEDDINGS_PATH = r"E:\Machine_learning\NLP\Project\Hybrid_search\artifacts\embeddings.npy"

def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    print(f"Načítám {MODEL_NAME} na {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 1024 # max delka tokenu co precte
    return model # vratim model co ma tokenizer, vahy NN, 1024 a encoder

def embed_articles(model, df):
    texts = (df["headline"] + "\n" + df["brief"] + "\n\n" + df["content"]).tolist()# vse se spoji a do listu
    return model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True) # prevedu na vektor space numpy matice (1000x1024), kazdy clanek je reprezentovan vektorem s 1024 cisly, 1000 radku 1024 cisel pro kazdy 

def load_embeddings(model, df):
    if os.path.exists(EMBEDDINGS_PATH): # pokud jsou embedd ulozene tak je nacte z disku pro rychlost
        return np.load(EMBEDDINGS_PATH) # matice (1000, 1024)
    embeddings = embed_articles(model, df) # vytvori matici (1000,1024) jeden clanek jeden vektor o 1024 cislech
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True) # vytvorim artifacts
    np.save(EMBEDDINGS_PATH, embeddings) # ulozi matici (1000,1024)
    return embeddings

def embed_queries(model, queries):
    #stejný model a normalizace jako u článků, jinak by vektory nešly porovnat
    return model.encode(queries, batch_size=32, normalize_embeddings=True)# otazka se taky prevede na matici 1 dotaz (1,1024). # (počet dotazů, 1024), pro 1 dotaz (1, 1024)

def dense_search(embeddings, query_vector, top_k=100):
    # kosinová podobnost dotazu se všemi články najednou (vektory mají délku 1, takže stačí skalární součin)
    scores = embeddings @ query_vector # (1000x1024) x (1024,) -> jedno cislo na clanek
    best = np.argsort(-scores)[:top_k] # vracim podobnost (100,) od nejmensi hodnoty, proto "-" at se to obrati od nejvetsi
    
    vysledek = []
    for i in best:              # projde každé doc_id v poli
        vysledek.append(int(i)) # převede numpy.int64 na obyčejný Python int a přidá do listu
    return vysledek
