# Hybrid Search – vyhledávání v českých zpravodajských článcích

Vyhledávač nad 1000 českými zpravodajskými články.

Řešení vznikalo ve třech iteracích:

- **A · BM25 (fulltextové vyhledávání).** Nejdříve jsem implementoval klasické
   vyhledávání BM25 včetně předzpracování češtiny (odstranění interpunkce, tokenizace, lemmatizace, malá písmena, odstranění diakritiky, odstranění stopslov, stemming).
- **B · BM25 + embeddingy (hybrid).** Na základě výsledků evaluace jsem projekt
   rozšířil o hybridní vyhledávání, které kombinuje BM25 s embeddingy modelu
   `BAAI/bge-m3`. Každý článek má jeden vektor. Výsledky obou metod se spojují pomocí
   Reciprocal Rank Fusion (RRF).
- **C · BM25 + embeddingy s chunkingem.** Embedding celého článku nevidí dobře na
   střed a konec dlouhých článků. Proto se články pro embeddingy rozdělí na chunky
   (max. 512 tokenů) a každý chunk má vlastní vektor. BM25 i RRF zůstávají stejné jako v B.

## Struktura projektu

| Soubor | Popis |
|---|---|
| `_data.ipynb` | průzkum a úprava datasetu |
| `preprocessing.py` | odstranění interpunkce, tokenizace, lemmatizace, malá písmena, odstranění diakritiky, odstranění stopslov, stemming |
| `bm25.py` | implementace BM25 (řešení A) |
| `embeddings.py` | embeddingy celých článků a dotazů (bge-m3) |
| `hybrid.py` | hybridní vyhledávání nad celými články (řešení B) |
| `chunker/chunking.py` | rozdělení článků na chunky (`SentenceSplitter` z LlamaIndex) |
| `chunker/embeddings.py` | embeddingy chunků a vyhledávání článků podle nejlepšího chunku |
| `chunker/hybrid.py` | hybridní vyhledávání s chunky (řešení C) |
| `main.py` | webová aplikace (FastAPI) s vyhledáváním BM25 |
| `eval_scripts/evaluate_bm25.py`, `eval_scripts/evaluate_hybrid.py` | LLM vygenerované pomocné skripty pro evaluaci na testovacích dotazech |
| `eval_data/` | testovací dotazy |


### Co se indexuje

Každý článek se pro vyhledávání skládá ze tří polí: `headline`, `brief` a `content`.

- **BM25:** pole se spojí mezerou (`headline + " " + brief + " " + content`) a poté
  projdou předzpracováním.
- **Embeddingy (bge-m3):** pole se spojí s oddělením řádků
  (`headline + "\n" + brief + "\n\n" + content`), aby model rozlišil titulek,
  brief a content.

Délka textu v tokenech modelu bge-m3: min 181, medián 786, průměr 915, max 5 308.
Nad 1024 tokenů má 29 % článků, nad 512 tokenů 78 %.

### Jak vyhledávání funguje

#### 1. Předzpracování textu (`preprocessing.py`)

Stejná funkce `preprocess()` se použije na články i na dotazy (jen pro BM25):

1. **Odstranění interpunkce**
2. **Tokenizace**
3. **Lemmatizace**
4. **Malá písmena**
5. **Odstranění diakritiky**
6. **Odstranění stopslov**
7. **Stemming**


#### 2. Indexace článků

- **BM25 (`bm25.py`)** – pro každý článek se uloží slovník četností slov
  `{slovo: počet}` a délka článku. Pro celý korpus se spočítá, v kolika článcích se
  každé slovo vyskytuje, a průměrná délka článku. Stejný index používají všechna tři
  řešení.

- **Embeddingy celých článků (`embeddings.py`, řešení B)** – každý článek se modelem
  `BAAI/bge-m3` převede na normalizovaný vektor o 1024 číslech. Vznikne hustá matice
  **1000 × 1024**, která se uloží do `artifacts/embeddings.npy`, aby se nemusela
  počítat znovu. Model čte maximálně 1024 tokenů; delší články se oříznou, díky
  pořadí headline → brief → content se ale vždy zachová titulek a brief.

- **Embeddingy chunků (`chunker/`, řešení C)** – `content` každého článku se rozdělí
  po větách na chunky o max. 512 tokenech (tokenizer bge-m3) s překryvem 64 tokenů.
  První chunk dostane titulek + brief + text, další chunky titulek + text, aby každý
  chunk věděl, o čem článek je. Z 1000 článků vznikne **2167 chunků**, každý má vlastní
  vektor. Matice **2167 × 1024** se uloží do `artifacts/embeddings_chunks.npy` a vedle
  ní `artifacts/chunk_doc_ids.npy`, které ke každému chunku říká, ze kterého článku je.
  Nic se neořezává.

#### 3. Vyhledávání

- **BM25** – dotaz projde předzpracováním a pro každý článek se spočítá BM25 skóre
  (`k1 = 1.5`, `b = 0.75`). Vrátí se články se skóre > 0 seřazené sestupně.
- **Embeddingy celých článků (B)** – dotaz se tím samým modelem převede na vektor
  (1024,). Kosinová podobnost se všemi články se spočítá jedním násobením matice
  `embeddings @ query_vector`.
- **Embeddingy chunků (C)** – vektor dotazu se porovná se všemi 2167 chunky. Článek
  dostane pořadí podle svého nejpodobnějšího chunku (MaxP) a každý článek se ve
  výsledku objeví jen jednou. Výstup je seznam `doc_id` ve stejném formátu jako v B.

#### 4. Hybrid – spojení výsledků (`hybrid.py`, `chunker/hybrid.py`)

1. Z BM25 i z embeddingů se vezme **100 nejlepších článků**.
2. Pořadí se spojí pomocí **Reciprocal Rank Fusion**:
   `skóre = 0.3 / (60 + pořadí v BM25) + 0.7 / (60 + pořadí v embeddingech)`.
   Spojují se pořadí, ne skóre, protože skóre BM25 a kosinová podobnost nejsou
   srovnatelné.
3. Výsledky se seřadí podle skóre a vrátí se top N.

#### Webová aplikace (`main.py`)

Aplikace nasazená na Azure používá kvůli nákladům jen BM25 (kroky 1–3 bez
embeddingů).

## Evaluace

Evaluace je typu **known-item**: každý testovací dotaz má přesně jeden správný článek
(`doc_id`) a měří se, na kolikátém místě ho vyhledávač vrátí v top 10.

- `eval_scripts/evaluate_bm25.py` – BM25 se spustí se 7 variantami předzpracování
  (kroky se postupně zapínají) a měří se přínos každého kroku.
  Výsledek: `eval_results/results_bm25.csv`.
- `eval_scripts/evaluate_hybrid.py` – srovnání tří řešení A, B a C na stejných dotazech
  a metrikách. Výsledky: `eval_results/results.csv`, `eval_results/per_query.csv`.

Dotazy i články procházejí vždy **stejnou** variantou předzpracování, jinak by se
tokeny dotazu a článku nemohly shodovat.

### Testovací sady (`eval_data/`)

| Soubor | Počet | Pole | Popis |
|---|---|---|---|
| `queries_known_item.json` | 240 | `query`, `doc_id`, `type` | ručně psané, přirozené dotazy |
| `queries_hard.json` | 500 | `query`, `doc_id`, `type` | těžké dotazy s malým překryvem slov s titulkem a briefem, jiné články než ruční sada |
| `queries_keywords.json` | 200 | `query`, `doc_id` | automaticky vytvořené z klíčových slov článků (sloupec `keywords`) |
| `queries_position.json` | 120 | `query`, `doc_id`, `type`, `token_pos` | dotazy na informace ze středu a konce 30 dlouhých článků (1500+ tokenů) |

Další sada **„ruční bez diakritiky“** se nenačítá ze souboru, vzniká v kódu ze sady
`queries_known_item.json` odstraněním diakritiky. Simuluje uživatele, který píše bez
háčků a čárek.

Ukázkový dotaz z každé sady:

| Sada | Příklad dotazu |
|---|---|
| ruční | „kdo převzal vedení PPF po smrti zakladatele“ |
| ruční bez diakritiky | „kdo prevzal vedeni PPF po smrti zakladatele“ |
| keywords | „Francie Prezidentské volby Marine le pen“ |
| těžké | „mattel prodává angličáky hot wheels jako nft za krypto“ |
| střed a konec | „katovi po popravě zůstala v ruce jen zrzavá paruka“ |

**Poznámka k původu dat:** Testovací sady byly vygenerovány pomocí LLM. Nejde tedy o přesné měření kvality vyhledávání v reálném provozu, ale o
orientační srovnání: slouží hlavně k tomu, abych viděl, jestli a jak moc se
vyhledávání zlepšuje mezi jednotlivými variantami (předzpracování, BM25, hybrid,
chunking). Absolutní hodnoty metrik je proto potřeba brát s rezervou.

### Typy dotazů (pole `type`)

| `type` | Co testuje | Příklady dotazů z testovacích sad |
|---|---|---|
| `lexical` | dotaz používá stejná slova jako článek | „výstava Josef Váchal“, „letiště Vodochody nebude“ |
| `short` | krátký dotaz (1–3 slova) | „Hřebeč“, „Bechný“ |
| `typo` | překlepy | „skoda enyak cena prodej“, „twiter klubhouse akvizice“ |
| `derived` | odvozená slova | „Avast fúzuje s Nortonem ekonom lituje“, „radilův návrat do pardubic z ruska“ |
| `synonym` | jiná slova se stejným významem | „rozvoz kanabisu přes Uber v Kanadě“, „útok střelce v dánském obchodním domě“ |
| `paraphrase` | celá myšlenka formulovaná jinak | „kdo převzal vedení PPF po smrti zakladatele“, „zesnul bývalý manžel špinarové, rakovina plic“ |
| `colloquial` | hovorová čeština | „proč jsou papriky tak drahý“, „lenny a týpek z linkinů spolu udělali singl“ |
| `question` | dotaz ve formě otázky (jen v těžké sadě) | „kde odehráli naši zápas s estonci místo tallinnu“, „hrozí po zákazu ruské ropy v Evropě nedostatek nafty?“ |
| `middle`, `end` | informace ze středu / konce dlouhého článku, se jmény a názvy z pasáže (jen v sadě střed a konec) | „katovi po popravě zůstala v ruce jen zrzavá paruka“ |
| `middle_paraphrase`, `end_paraphrase` | stejná pasáž popsaná jinými slovy (jen v sadě střed a konec) | – |

### Metriky

Pro každý dotaz se spočítají metriky z pozice správného článku v top 10 a pak se
zprůměrují přes celou sadu. Příklad: správný článek je na **3. místě**.

| Metrika | Výpočet | Příklad (3. místo) | Význam |
|---|---|---|---|
| **MRR@10** | 1 / pořadí | 0,33 | hlavní metrika, trestá každé místo pod prvním |
| **R@1** | je na 1. místě? | 0 | trefa hned napoprvé |
| **R@5** | je v top 5? | 1 | uživatel článek uvidí bez scrollování |
| **R@10** | je v top 10? | 1 | článek se vůbec našel |
| **nDCG@10** | 1 / log₂(pořadí + 1) | 0,5 | jako MRR, ale mírnější trest za nižší místa |

Když správný článek v top 10 není, jsou všechny metriky dotazu 0. U sad s polem
`type` se MRR@10 počítá navíc zvlášť pro každý typ dotazu.

### Výsledky BM25 – varianty předzpracování

Varianty (kroky se postupně přidávají):

| Varianta | Předzpracování |
|---|---|
| 0 baseline | jen odstranění interpunkce a tokenizace |
| 1 + malá písmena | + převod na malá písmena |
| 2 + bez diakritiky | + odstranění diakritiky |
| 3 + stopwords | + odstranění stopslov |
| 4 + lemmatizace | + lemmatizace |
| 5 + stemming (full) | + stemming (= `preprocess()` v projektu) |
| 6 stemming bez lemmatizace | jako 5, ale bez lemmatizace |

MRR@10 podle sady:

| Varianta | ruční | ruční bez diakritiky | keywords | těžké (500) |
|---|---|---|---|---|
| 0 baseline | 0.738 | 0.477 | 0.537 | 0.418 |
| 1 + malá písmena | 0.767 | 0.498 | 0.718 | 0.632 |
| 2 + bez diakritiky | 0.802 | 0.802 | 0.723 | 0.669 |
| 3 + stopwords | 0.804 | 0.804 | 0.723 | 0.665 |
| 4 + lemmatizace | 0.890 | 0.792 | 0.839 | 0.749 |
| **5 + stemming (full)** | **0.923** | 0.899 | **0.848** | **0.806** |
| 6 stemming bez lemmatizace | 0.906 | **0.906** | 0.846 | 0.787 |

MRR@10 podle typu dotazu, baseline vs. plné předzpracování:

| `type` | ruční: baseline | ruční: full | těžké: baseline | těžké: full |
|---|---|---|---|---|
| lexical | 0.921 | 0.971 | – | – |
| short | 0.911 | 1.000 | – | – |
| typo | 0.423 | 1.000 | 0.196 | 0.893 |
| derived | 0.762 | 0.960 | 0.414 | 0.887 |
| colloquial | 0.623 | 0.875 | 0.320 | 0.748 |
| paraphrase | 0.785 | 0.902 | 0.476 | 0.804 |
| synonym | 0.666 | 0.840 | 0.513 | 0.757 |
| question | – | – | 0.521 | 0.815 |

Hlavní zjištění:

- **Plné předzpracování (varianta 5) je nejlepší téměř všude.** Oproti baseline zvýší
  MRR@10 o +0.19 (ruční), +0.42 (bez diakritiky), +0.31 (keywords) a +0.39 (těžké).
- **Převod na malá písmena** výrazně pomůže u keywords (0.537 → 0.718) a těžkých
  dotazů (0.418 → 0.632), kde jsou jména často psaná malými písmeny, zatímco
  v článcích velkými.
- **Odstranění diakritiky** má největší efekt na dotazy bez háčků a čárek
  (0.498 → 0.802).
- **Odstranění stopslov** nemá téměř žádný vliv (změny do ±0.004). BM25 časté slova
  sám potlačuje přes IDF.
- **Lemmatizace** je největší jednotlivé zlepšení u ručních dotazů (0.804 → 0.890)
  a pomáhá hlavně u odvozených tvarů slov. U dotazů bez diakritiky ale mírně
  škodí (0.804 → 0.792), protože lemmatizátor slova bez háčků nepozná. To vyrovná
  až stemming.
- **Stemming** přidává další zlepšení nad lemmatizací (ruční 0.890 → 0.923, těžké
  0.749 → 0.806). S plným předzpracováním dosáhnou ruční `typo` 1.000 a `derived`
  0.960 (baseline 0.423 a 0.762).
- **Stemming bez lemmatizace (varianta 6)** je skoro stejně dobrý, na dotazech bez
  diakritiky dokonce o kousek lepší, ale na těžké sadě horší (0.787 vs. 0.806). Proto
  projekt používá variantu 5.
- **Slabina BM25 zůstává u dotazů jinými slovy**: synonyma (0.840 / 0.757),
  hovorová čeština a parafráze v těžké sadě. BM25 umí najít jen shodná slova, ne
  shodný význam. To byl důvod pro rozšíření o hybridní vyhledávání s embeddingy.

### Výsledky: A vs. B vs. C

Srovnávaná řešení:

- **A · BM25 (full)** – BM25 s plným předzpracováním.
- **B · hybrid, dense 0.7** – BM25 + embeddingy celých článků spojené přes RRF
  (váha embeddingů 0.7, BM25 0.3).
- **C · hybrid + chunky, dense 0.7** – stejné jako B, ale embeddingy se počítají
  nad chunky (max. 512 tokenů) a článek dostane pořadí svého nejlepšího chunku.

MRR@10 podle sady (průměr = MRR@10 přes 5 sad):

| Řešení | ruční | ruční bez diakritiky | keywords | těžké (500) | střed a konec (120) | průměr |
|---|---|---|---|---|---|---|
| A · BM25 (full) | 0.923 | 0.899 | **0.848** | 0.806 | 0.810 | 0.857 |
| B · hybrid, dense 0.7 | **0.977** | **0.961** | 0.787 | **0.933** | 0.754 | 0.882 |
| **C · hybrid + chunky, dense 0.7** | 0.972 | 0.954 | 0.782 | 0.926 | **0.904** | **0.908** |

Všechny metriky, průměr přes 5 sad:

| Řešení | MRR@10 | R@1 | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| A · BM25 (full) | 0.857 | 0.804 | 0.930 | 0.950 | 0.880 |
| B · hybrid, dense 0.7 | 0.882 | 0.834 | 0.947 | 0.964 | 0.903 |
| **C · hybrid + chunky, dense 0.7** | **0.908** | **0.861** | **0.965** | **0.979** | **0.925** |



#### Hlavní zjištění

- **Hybrid (B) výrazně zlepší BM25 u přirozených a těžkých dotazů** (těžká sada
  0.806 → 0.933). Embeddingy rozumí významu, takže zvládají synonyma, parafráze
  i hovorovou češtinu, kde BM25 selhává.
- **U dotazů z klíčových slov zůstává nejlepší BM25** (0.848 vs. 0.787 u B). Seznam
  jmen a pojmů bez souvislé věty embeddingy zachytí hůř než přesná shoda slov.
- **B je slabé na dlouhých článcích.** Na sadě střed a konec je dokonce horší než
  samotné BM25 (0.754 vs. 0.810), protože embedding celého článku se ořízne na
  1024 tokenů a konec dlouhého textu v něm zanikne.
- **Chunking (C) tuto slabinu odstraní**: na sadě střed a konec stoupne MRR@10
  z 0.754 na 0.904, nejvíc u parafrázovaných pasáží. Na ostatních sadách jsou B a C
  prakticky stejné (rozdíly do 0.01).
- **Pořadí A < B < C platí ve všech metrikách** (MRR, R@1, R@5, R@10, nDCG), takže
  závěr nezávisí na volbě metriky. S řešením C je správný článek na 1. místě
  v 86 % dotazů (BM25 80 %) a v top 10 v 98 % dotazů (BM25 95 %).
- **Velikost chunku** 256 místo 512 tokenů výsledky prakticky nemění (průměr 0.912
  vs. 0.908), 512 dává menší index a rychlejší kódování.
- **Závěr:** nejlepší je řešení C (průměr 0.908), oproti BM25 je lepší na 4 z 5 sad.
  Webová aplikace nasazená na Azure používá kvůli nákladům jen
  BM25 (bez GPU a 2+ GB modelu), hybrid běží lokálně.

