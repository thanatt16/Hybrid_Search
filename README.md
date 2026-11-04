# Hybrid Search – vyhledávání v českých zpravodajských článcích

Vyhledávač nad 1000 českými zpravodajskými články.

Řešení vznikalo ve dvou fázích:

1. **BM25 (fulltextové vyhledávání).** Nejdříve jsem implementoval klasické
   vyhledávání BM25 včetně předzpracování češtiny (odstranění interpunkce, tokenizace, lemmatizace, malá písmena, odstranění diakritiky, odstranění stopslov, stemming)
2. **Hybridní vyhledávání.** Na základě výsledků evaluačních metrik jsem projekt
   rozšířil o hybridní vyhledávání, které kombinuje BM25 s embeddingy modelu
   `BAAI/bge-m3`. Výsledky obou metod se spojují pomocí Reciprocal Rank Fusion (RRF).

## Struktura projektu

| Soubor | Popis |
|---|---|
| `_data.ipynb` | průzkum a úprava datasetu |
| `preprocessing.py` | odstranění interpunkce, tokenizace, lemmatizace, malá písmena, odstranění diakritiky, odstranění stopslov, stemming |
| `bm25.py` | implementace BM25 |
| `embeddings.py` | embeddingy článků a dotazů (bge-m3) |
| `hybrid.py` | hybridní vyhledávání (BM25 + embeddingy, RRF) |
| `main.py` | webová aplikace (FastAPI) s vyhledáváním BM25 |
| `evaluate_bm25.py`, `evaluate_hybrid.py` | LLM vygenerované pomocné skripty pro evaluaci na testovacích dotazech |
| `eval/` | testovací dotazy |
| `results/` | výsledky evaluace |

