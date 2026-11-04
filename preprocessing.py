import pandas as pd
import simplemma

s_diakritikou  = "áčďéěíňóřšťúůýž"
bez_diakritiky = "acdeeinorstuuyz"

stopwords = {
    "a", "aby", "ale", "ani", "asi", "az", "bez", "co", "ci", "do", "jak", "jako",
    "jeho", "jeji", "jejich","jejimz", "jen", "jeste", "jiz", "k", "kde", "kdy", "kdyz",
    "ktery", "ktera", "ktere", "mezi", "na", "nad", "ne", "nebo", "nez", "o", "od",
    "po", "pod", "podle", "pro", "proti", "pred", "pri", "s", "se", "si", "tak",
    "take", "tam", "ten", "ta", "to", "tu", "u", "uz", "v", "ve", "vsak", "z", "za",
    "ze", "i", "by", "je", "jsou", "byl", "byla", "bylo", "byli", "jsem", "bude",
    "budou", "jsme", "jste", "on", "ja", "ty", "my", "vy", "svuj", "jenz", "tento", "dalsi", "hodne",
    "vsechen", "jeden", "mit", "moci", "muset", "chtit", "uvest","pak", "rad", "muj", "rici", "ted",
}

suffixes = ["atech", "atum", "etem", "ach", "ama", "ami", "ata", "aty", "ech", "emi", "emu", "ich",
            "imu", "ove", "ovi", "ych", "ymu", "ej", "em", "es", "im", "ou", "um", "ym",
            "a", "e", "i", "o", "u", "y"]


def remove_punctuation(text):
    clean = ""
    for ch in text:
        #print(ch,ch.isalnum())
        if ch.isalnum():
            clean = clean + ch
        else:
            clean = clean + " "
    return clean


def tokenize(text):
    tokens = text.split()
    return tokens

def lemmatize(tokens):
    lemmatized = []
    for t in tokens:
        lemmatized.append(simplemma.lemmatize(t, lang="cs"))
    return lemmatized

def lowercase(tokens):
    lowercased = []
    for t in tokens:
        lowercased.append(t.lower())
    return lowercased

def strip_diacritics(word):
    for a, b in zip(s_diakritikou, bez_diakritiky):
        word = word.replace(a, b)
    return word

def remove_stopwords(tokens):
    filtered_tokens = []
    for word in tokens:
        if word not in stopwords:
            filtered_tokens.append(word)
    return filtered_tokens

def stem(word):
    for suffix in suffixes:
        if word.endswith(suffix) and (len(word) - len(suffix)) >= 3:
            word = word[:-len(suffix)]
            break
    return word

def preprocess(text, lemma=True, stemming=True): 
    text = remove_punctuation(text)
    tokens = tokenize(text)
    if lemma:
        tokens = lemmatize(tokens)
        tokens = [t for t in tokens if t != "být"]
    tokens = lowercase(tokens)
    tokens = [strip_diacritics(t) for t in tokens]
    tokens = remove_stopwords(tokens)
    
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens

if __name__ == "__main__":
    df = pd.read_parquet(r"E:\Machine_learning\NLP\Project\Hybrid_search\dataset\articles.parquet")
    texts = df["headline"] + " " + df["brief"] + " " + df["content"]

    all_tokens = []
    for text in texts:
        tokens = preprocess(text)
        all_tokens.append(tokens)

    counts = {}
    for clanek in all_tokens:
        for token in clanek:
            counts[token] = counts.get(token, 0) + 1
    serazene = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(serazene[:20])   # 20 nejčastějších tokenů
    print(serazene[-20:])  
