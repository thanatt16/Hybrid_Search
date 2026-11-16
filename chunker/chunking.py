import numpy as np
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer


def make_splitter():
    # chunk_size se pocita tokenizerem bge-m3, ne vychozim tokenizerem LlamaIndexu (OpenAI), jinak by 512 neodpovidalo modelu
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return SentenceSplitter(chunk_size=256,chunk_overlap=64,tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
)

def chunk_articles(df):
    splitter = make_splitter()
    chunks = []          # text chunku
    chunk_doc_ids = []   # ke kazdemu chunku doc_id clanku, ze ktereho pochazi

    for doc_id, (headline, brief, content) in enumerate(zip(df["headline"], df["brief"], df["content"])):
        parts = splitter.split_text(content) or [""]   # content rozdeli po vetach; prazdny content -> aspon titulek a perex
        for j, part in enumerate(parts):
            # prvni chunk dostane titulek i brief, dalsi jen titulek, aby kazdy chunk vedel, o cem clanek je
            prefix = headline + "\n" + brief + "\n\n" if j == 0 else headline + "\n"
            chunks.append(prefix + part)
            chunk_doc_ids.append(doc_id)

    return chunks, np.array(chunk_doc_ids)   # 1000 clanku -> cca 2167 chunku


















if __name__ == "__main__":
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    chunks, chunk_doc_ids = chunk_articles(df)

    print(f"{len(df)} clanku -> {len(chunks)} chunku")
    print("chunku na clanek:", pd.Series(chunk_doc_ids).value_counts().value_counts().sort_index().to_dict())

    # ukazka prvnich 3 chunku nejdelsiho clanku
    doc_id = int(df["content"].str.len().idxmax())
    for i in np.where(chunk_doc_ids == doc_id)[0][:3]:
        print(f"\n--- chunk {i} (clanek {doc_id}) ---")
        print(chunks[i][:300])
