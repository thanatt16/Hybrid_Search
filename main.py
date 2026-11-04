from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import html

from pathlib import Path
import pandas as pd

from bm25 import build_index, search
from preprocessing import preprocess

DATA_PATH = Path(__file__).parent / "dataset" / "articles.parquet"

df = pd.read_parquet(DATA_PATH)
texts = df["headline"] + " " + df["brief"] + " " + df["content"]
index = build_index([preprocess(text) for text in texts])
print("Index hotový, článků:", len(df))

app = FastAPI()


def get_results(q, top_n=5):
    """Vyhledá dotaz a vrátí seznam výsledků jako slovníky."""
    answer = search(q, index, top_n=top_n)
    results = []
    for poradi, (i, score_value) in enumerate(answer, start=1):
        results.append({
            "poradi": poradi,
            "score": round(score_value, 4),
            "headline": df["headline"].iloc[i],
            "url": df["url"].iloc[i],
        })
    return results


@app.get("/", response_class=HTMLResponse)
def home(q: str | None = None):
    results_html = ""

    if q:
        results = get_results(q, top_n=5)
        if not results:
            results_html = "<p>Nic nenalezeno.</p>"
        for r in results:
            headline = html.escape(r["headline"])
            url = html.escape(r["url"])
            results_html += f"""
            <div class="result">
                <p>{r["poradi"]}. [{r["score"]:.4f}] {headline}</p>
                <p>Odkaz zde -> <a href="{url}" target="_blank">{url}</a></p>
            </div>
            """

    value = html.escape(q or "")

    return f"""
    <!doctype html>
    <html lang="cs">
    <head>
        <meta charset="utf-8">
        <title>Hledání v článcích</title>
        <style>
            body {{
                font-family: sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-top: 15vh;
            }}
            input {{ width: 400px; padding: 10px; font-size: 18px; }}
            .result {{ width: 600px; border-bottom: 1px solid #ddd; padding: 8px 0; }}
        </style>
    </head>
    <body>
        <h1>Hledat v článcích</h1>
        <form action="/" method="get">
            <input name="q" value="{value}" placeholder="např. citibank" autofocus>
        </form>
        {results_html}
    </body>
    </html>
    """


@app.get("/search")
def search_endpoint(q: str, top_n: int = 5):
    return get_results(q, top_n)