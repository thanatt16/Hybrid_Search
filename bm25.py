from preprocessing import preprocess
import pandas as pd
import math

def count_words(tokens):
    word_count = {}
    for token in tokens:
        word_count[token] = word_count.get(token, 0) + 1
    return word_count

def build_index(corpus):
    docs = []
    docs_lengths = []

    for tokens in corpus:
        docs.append(count_words(tokens))
        docs_lengths.append(len(tokens))

    #print(f"Number of documents: {len(docs)}")
    #print(docs_lengths[0])
    #print(docs[0].get("soud", 0))

    avgdl = sum(docs_lengths) / len(docs_lengths)
    #print(f"Average document length: {avgdl:.2f}")

    doc_freqs = {}
    for doc in docs:
        for word in doc:
            doc_freqs[word] = doc_freqs.get(word, 0) + 1
    #print(len(doc_freqs))
    #print(doc_freqs.get("soud", 0))

    return {"docs": docs, "docs_lengths": docs_lengths, "avgdl": avgdl, "doc_freqs": doc_freqs}

def IDF(slovo, doc_freqs,N):
    n = doc_freqs.get(slovo, 0)
    return math.log((N - n + 0.5) / (n + 0.5) + 1)
#print(IDF("soud", doc_freqs, len(docs)))

k1 = 1.5
b = 0.75

def score(query, doc, doc_length, avgdl, doc_freqs, N):
    score = 0.0
    for word in query:
        if word in doc:
            idf = IDF(word, doc_freqs, N)
            tf = doc[word]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))
    return score
#preprocessed_query = preprocess("hypotéka úrokové sazby")
#print(preprocessed_query)
#print(score(preprocessed_query, docs[0], docs_lengths[0], avgdl, doc_freqs, len(docs)))

def search_tokens(preprocessed_query, index, top_n=5):
    docs = index["docs"]
    vysledky = []
    for i, doc in enumerate(docs):
        doc_length = index["docs_lengths"][i]
        score_value = score(preprocessed_query, doc, doc_length, index["avgdl"], index["doc_freqs"], len(docs))
        if score_value > 0:
            vysledky.append((i, score_value))

    vysledky.sort(key=lambda x: x[1], reverse=True)
    return vysledky[:top_n]

def search(query, index, top_n=5):
    return search_tokens(preprocess(query), index, top_n)


if __name__ == "__main__":
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]
    index = build_index([preprocess(text) for text in texts])

    while True:
        query = input("\nHledat: ")
        answer = search(query, index, top_n=5)
        if not answer:
            print("Nic nenalezeno.")
        for poradi, (i, score_value) in enumerate(answer, start=1):
            print(f"{poradi}. [{score_value:.2f}] {df['headline'].iloc[i]}")
            print(f"Odkaz zde -> {df['url'].iloc[i]}")
