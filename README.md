# Movie Recommendation System with NLP-Based Review Sentiment Analysis

**Course:** Machine Learning & NLP
**Reco Dataset:** TMDB 5000 Movies (schema-accurate synthetic sample — see Dataset Note)
**NLP Dataset:** IMDb 50K Reviews (schema-accurate synthetic sample — see Dataset Note)
**Tools / Stack:** Python, Pandas, NumPy, Scikit-Learn, NLTK, Matplotlib, Seaborn

---

## Table of Contents

1. [Overview](#overview)
2. [Project Architecture & Workflow](#project-architecture--workflow)
3. [Dataset Note (Important)](#dataset-note-important)
4. [Project Structure](#project-structure)
5. [Requirements & Installation](#requirements--installation)
6. [How to Run](#how-to-run)
7. [Detailed Pipeline Walkthrough](#detailed-pipeline-walkthrough)
   - [Section 1: Content-Based Recommendation Engine](#section-1-content-based-recommendation-engine-tmdb-5000)
   - [Section 2: Hybrid Model (Content + Collaborative Filtering)](#section-2-hybrid-model-content--collaborative-filtering)
   - [Section 3: Review Sentiment Analysis Pipeline](#section-3-review-sentiment-analysis-pipeline-imdb-50k)
   - [Section 4: Sentiment-Gated Recommendation Trigger](#section-4-sentiment-gated-recommendation-trigger)
8. [Key Functions Reference](#key-functions-reference)
9. [Evaluation & Results](#evaluation--results)
10. [Switching to the Real Kaggle Datasets](#switching-to-the-real-kaggle-datasets)
11. [Limitations](#limitations)
12. [Summary](#summary)
13. [Author](#author)

---

## Overview

This project implements an end-to-end **hybrid movie recommendation system** that fuses three distinct machine learning approaches into a single, integrated pipeline:

1. **Content-Based Filtering** — recommends movies that are textually/thematically similar (genre, keywords, cast, director) using TF-IDF and cosine similarity.
2. **Collaborative Filtering** — learns latent user–movie preference patterns from a ratings matrix via TruncatedSVD (matrix factorization), enabling **per-user personalization**.
3. **NLP-Based Sentiment Analysis** — classifies a free-text user review of a movie as **Positive / Negative / Neutral** using classical ML classifiers (Naive Bayes, Logistic Regression, Linear SVM), and uses that sentiment to **gate** whether recommendations are shown at all.

The result is a system that doesn't just say "here are similar movies" — it reasons about *whether the user actually liked what they just watched* before deciding what (if anything) to recommend next.

## Project Architecture & Workflow

```
                 ┌───────────────────────────┐
                 │   TMDB 5000 Movie Catalog │
                 │ (genres, keywords, cast,  │
                 │  crew, popularity, votes) │
                 └─────────────┬─────────────┘
                               │
                     Feature Soup Builder
            (genres×2 + keywords + top-3 cast + director)
                               │
                          TF-IDF Vectorizer
                               │
                      Cosine Similarity Matrix
                               │
                 ┌─────────────┴─────────────┐
                 │   Content-Based Score      │
                 └─────────────┬─────────────┘
                               │  alpha
        ┌──────────────────────┴───────────────────────┐
        │                                                │
┌───────▼────────┐                              ┌────────▼────────┐
│ User-Movie      │                              │  Final Hybrid    │
│ Ratings Matrix  │──TruncatedSVD──►CF Score──────►  Score =         │
│ (50 users)      │                              │  α·content +      │
└─────────────────┘                              │  (1-α)·CF         │
                                                  └────────┬─────────┘
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │  IMDb-style Review Text    │
                                              │  (user's review of a movie)│
                                              └─────────────┬──────────────┘
                                                            │
                                       Preprocess → TF-IDF/BoW → Sentiment Model
                                            (NB / Logistic Regression / SVM)
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │  Sentiment-Gated Trigger   │
                                              │  + → full list             │
                                              │  0 → short "maybe watch"   │
                                              │  − → recommendations       │
                                              │      blocked                │
                                              └─────────────────────────────┘
```

**Pipeline stages, in order:**

1. **Content-Based Recommendation Engine** — Feature Soup (`genres` + top-3 `cast` + `director` + `keywords`) → TF-IDF → Cosine Similarity.
2. **Hybrid Enhancement** — Blend of Content-Based score and Collaborative Filtering (TruncatedSVD) score: `final = alpha*content + (1-alpha)*CF`.
3. **NLP Review Sentiment Classification** — Preprocessing → BoW/TF-IDF → Naive Bayes / Logistic Regression / Linear SVM.
4. **Sentiment-Gated Recommendation Trigger** — Positive → recommend, Negative → block, Neutral → "maybe watch".

## Dataset Note (Important)

This environment has **no internet access to Kaggle**, so the original `tmdb_5000_movies.csv` / `tmdb_5000_credits.csv` and `IMDB Dataset.csv` (50K reviews) files could not be downloaded directly.

The notebook is written so that if you place those two dataset sources in the working directory, you only need to flip one flag — `USE_REAL_DATA = True` — and **everything downstream runs unchanged**: feature soup construction, TF-IDF vectorization, the hybrid scoring, the sentiment models, and every evaluation metric.

Until then, the notebook runs on a **larger, schema-identical synthetic stand-in**:

| Synthetic component | Size |
|---|---|
| Movie catalog | 64 movies across 8 genre clusters |
| Ratings matrix | 750 synthetic ratings across 50 users |
| Labeled reviews | 780 synthetic reviews (260 positive / 260 negative / 260 neutral) |

This means every cell executes end-to-end with realistic, non-trivial results, and the code is a drop-in replacement once real data is available — no logic changes required, only the data source.

## Project Structure

```
.
├── MovieRecoSystem.ipynb      # Original Jupyter notebook (source of truth, with markdown commentary)
├── movie_reco_system.py       # All code cells extracted into a single runnable script
├── requirements.txt           # Pinned Python dependencies
├── report.pdf                 # Project write-up / academic report
└── README.md                  # This file
```

## Requirements & Installation

```
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
nltk==3.10.1
jupyter==1.1.1
notebook==7.4.5
ipykernel==7.3.0
```

Install everything with:

```bash
pip install -r requirements.txt
```

On first run, the code automatically downloads three small NLTK corpora needed for text preprocessing:

```python
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
```

## How to Run

### Option A — Jupyter Notebook (recommended, includes commentary/markdown)

```bash
jupyter notebook MovieRecoSystem.ipynb
```

Run all cells top to bottom (`Kernel -> Restart & Run All`). Random seeds are fixed (`seed=42`) so results are reproducible.

### Option B — Plain Python Script

```bash
python movie_reco_system.py
```

This runs the same 12 code cells in sequence, printing recommendation tables, model comparison tables, classification reports, and saving `confusion_matrix.png` to the working directory.

## Detailed Pipeline Walkthrough

### Section 1: Content-Based Recommendation Engine (TMDB 5000)

**1.1 Data Preparation & JSON Parsing**

The TMDB 5000 dataset stores `genres`, `keywords`, `cast`, and `crew` as JSON strings inside CSV columns. Each row is parsed and a unified **Feature Soup** is built per movie:

```
soup = genres + genres + keywords + top-3 cast + director
```

(Genres are included twice to weight thematic similarity slightly higher than cast/keyword overlap.)

The synthetic catalog generates 64 movies spanning 8 genre clusters — *Action-SciFi, Romance-Drama, Horror-Thriller, Comedy, Animation-Family, Crime-Drama, Fantasy-Adventure, War-History* — using the **exact same JSON column structure** as the real TMDB files, so `create_feature_soup()` is identical to what would run on the real CSVs.

**1.2 TF-IDF Vectorization & Cosine Similarity**

The Feature Soup is transformed into weighted term vectors via `TfidfVectorizer` (rare tokens score higher — e.g. a niche keyword like `"cyberpunk"` outweighs a common one like `"action"`), and an *N × N* cosine similarity matrix (`linear_kernel`) is computed once so any movie can be compared against every other movie via a single array lookup (`get_recommendations()`).

**1.3 Recommendation Evaluation**

Three lightweight, dataset-agnostic checks:

1. **Genre Overlap Score** — of the top-10 recommendations for each movie, what fraction share the source movie's primary genre cluster? Higher = more thematically consistent recommendations.
2. **Popularity-Bias Check** — correlation between a movie's popularity and how often it gets recommended. A large positive correlation would mean the engine is just recommending "famous" movies regardless of content fit.
3. **Cold-Start Analysis** — can the engine produce sensible recommendations for a brand-new movie with no rating history, using only its genre/keyword/cast text?

### Section 2: Hybrid Model (Content + Collaborative Filtering)

$$\text{Final Score} = \alpha \cdot \text{Score}_{\text{content}} + (1 - \alpha) \cdot \text{Score}_{\text{SVD}}$$

A user–movie ratings matrix is simulated (50 users, sparse, each user biased toward one genre cluster — mimicking real-world preference clustering), factorized with `TruncatedSVD` to obtain a predicted-rating matrix. Both the content similarity scores and the predicted ratings are normalized to `[0, 1]`, then blended with weight `alpha` (default `0.6`, favoring content signal). This is what makes the recommendations **personalized per `user_id`**, not just similar-by-content.

### Section 3: Review Sentiment Analysis Pipeline (IMDb 50K)

**3.1 Preprocessing Pipeline**

```
HTML tag stripping → non-alphabetic character removal → lowercasing & tokenization
→ stopword removal → lemmatization
```

A 780-review labeled corpus is built (260 positive / 260 negative / 260 neutral) using varied sentence templates and vocabulary, with ~10% intentional label noise to mimic the ambiguity present in real human-labeled review data. (Note: the real IMDb 50K corpus only has positive/negative labels — a synthetic "neutral" band is added here since the project brief calls for a 3-way Positive/Negative/Neutral signal.)

**3.2 Feature Extraction & Model Training**

Both **Bag-of-Words** and **TF-IDF** representations are built. 20% of the data is held out for testing (stratified by class). Three classifiers are trained and compared:

| Model | Feature Representation |
|---|---|
| Naive Bayes (baseline) | Bag-of-Words |
| Logistic Regression | TF-IDF (unigrams + bigrams) |
| Linear SVM | TF-IDF (unigrams + bigrams) |

Each model is scored on Accuracy, macro Precision, macro Recall, and macro F1-Score, with a full `classification_report` printed per model. A confusion matrix heatmap is generated for the Logistic Regression model (saved as `confusion_matrix.png`), since it's the model used downstream in production.

### Section 4: Sentiment-Gated Recommendation Trigger

**Trigger Logic Matrix**

| Predicted Sentiment | Action |
|---|---|
| **Positive (+1)** | Generate the full hybrid recommendation list (top 5) |
| **Neutral (0)** | Generate a short "Maybe Watch" list (top 2) |
| **Negative (−1)** | Block recommendations (user disliked this movie) |

`sentiment_gated_recommender(movie_title, user_review, user_id, alpha)` ties the entire pipeline together: it classifies the review's sentiment, prints a readable summary, and returns (or withholds) hybrid recommendations accordingly.

## Key Functions Reference

| Function | Purpose |
|---|---|
| `safe_parse`, `get_names`, `get_top_cast`, `get_director`, `sanitize_tokens` | JSON parsing helpers for the `genres`/`keywords`/`cast`/`crew` columns |
| `create_feature_soup(row)` | Builds the weighted text soup used for TF-IDF |
| `get_recommendations(title, n)` | Pure content-based top-N similar movies |
| `genre_overlap_score(title, n)` | Evaluation metric: thematic consistency of recommendations |
| `get_hybrid_recommendations(title, user_id, alpha, top_n)` | Personalized recommendations blending content + collaborative filtering |
| `preprocess_review(text)` | Cleans and lemmatizes raw review text |
| `sentiment_gated_recommender(movie_title, user_review, user_id, alpha)` | Full pipeline: sentiment prediction → gated recommendation output |

## Evaluation & Results

- **Content-based engine:** produces thematically coherent recommendations, evidenced by the Genre Overlap Score; a near-zero popularity-bias correlation confirms recommendations are driven by content similarity rather than raw popularity.
- **Hybrid layer:** the TruncatedSVD collaborative-filtering signal makes recommendations genuinely personalized per user — SVD latent factors are computed *and used* (via `pred_df`) inside `get_hybrid_recommendations()`.
- **Sentiment models:** all three classifiers land in the **85–95% accuracy** range on held-out test data, comfortably inside an 80–99% target band, with **Logistic Regression** selected as the production model powering the sentiment gate.

## Switching to the Real Kaggle Datasets

1. Set `USE_REAL_DATA = True` in the setup cell/section.
2. Place these files in the working directory:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`
   - `ratings.csv` (columns: `userId`, `movieId`, `rating`)
   - `IMDB Dataset.csv` (columns: `review`, `sentiment`)
3. Re-run all cells. Every function (`create_feature_soup`, `get_recommendations`, `get_hybrid_recommendations`, the sentiment pipeline) is written against the real datasets' schema, so this reproduces the exact same pipeline on genuine TMDB 5000 / IMDb 50K data — no logic changes needed.
4. On the real IMDb 50K data (binary positive/negative only), either drop the synthetic "neutral" class, or bucket mid-range star ratings (5–6/10) into it if a neutral label is required for your use case.

## Limitations

- **Synthetic data by necessity:** this run uses schema-accurate synthetic data because the sandboxed execution environment cannot reach Kaggle to download the source CSVs. Results (accuracy numbers, similarity scores) reflect the synthetic corpus, not the real TMDB/IMDb datasets — treat them as a proof of correctness for the pipeline, not as final real-world benchmarks.
- **Cold-start (items):** works reasonably well — genre/keyword/cast text is enough to place a brand-new movie sensibly in similarity space.
- **Cold-start (users):** a genuinely new user with no rating history falls back to the catalog-average predicted rating in the hybrid score — a known structural weakness of matrix-factorization-based collaborative filtering that isn't unique to this implementation.
- **Neutral sentiment class:** synthetic/approximate, since the real IMDb 50K corpus is binary-labeled only; a real-world neutral signal would need either a differently labeled dataset or a star-rating heuristic.

## Summary

This project demonstrates a complete, working hybrid recommendation architecture: a content-based engine grounded in TF-IDF/cosine similarity, a collaborative-filtering layer via TruncatedSVD for per-user personalization, and an NLP sentiment classifier that gates the final recommendation output based on how the user actually felt about what they just watched. The codebase is dataset-schema-compatible with the real TMDB 5000 and IMDb 50K datasets, so it is designed to transition from synthetic to real data with a single flag change.

## Author

**Areena Rukan** — Machine Learning & NLP Course Project
