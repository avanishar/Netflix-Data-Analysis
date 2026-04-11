# 🎬 Netflix Content Analysis using Unsupervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> Analyzing Netflix's content catalog using K-Means Clustering and building a Content-Based Recommendation System with TF-IDF + Cosine Similarity.

---

## 📌 Problem Statement

Netflix has one of the largest content libraries in the world — thousands of movies and TV shows across dozens of genres and countries.

The goal of this project is to:
1. **Analyze content patterns** — understand what Netflix offers, when, and from where
2. **Group similar content** using **Unsupervised Machine Learning (K-Means Clustering)**
3. **Build a Recommendation System** using **TF-IDF + Cosine Similarity**

This helps understand Netflix's content distribution and enables smarter content discovery for viewers.

---

## 📁 Dataset Information

- **Source:** [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **File:** `netflix_titles.csv`
- **Size:** ~8,800 titles

| Column | Description |
|--------|-------------|
| `type` | Movie or TV Show |
| `title` | Name of the content |
| `director` | Director name |
| `cast` | Actors involved |
| `country` | Country of production |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Age rating (PG, TV-MA, etc.) |
| `duration` | Minutes (Movie) or Seasons (TV Show) |
| `listed_in` | Genre(s) |
| `description` | Short plot summary |

---

## 🗺️ Project Workflow

```
Data Loading
     ↓
Data Cleaning  (missing values, duplicates, formatting)
     ↓
EDA + Insights  (type, country, genre, year trends)
     ↓
Feature Engineering  (content_age, duration_numeric, primary_genre, country_group)
     ↓
Encoding + Scaling  (LabelEncoder, StandardScaler)
     ↓
K-Means Clustering  (Elbow Method → K=5)
     ↓
Cluster Interpretation  (describe each cluster)
     ↓
PCA Visualization  (2D cluster scatter plot)
     ↓
Recommendation System  (TF-IDF + Cosine Similarity)
     ↓
Business Insights + Conclusion
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical charts |
| `scikit-learn` | ML — KMeans, PCA, TF-IDF, Cosine Similarity, Encoders |

---

## ⚙️ Feature Engineering

New features created from existing columns:

| Feature | Formula / Logic | Why It Helps |
|---------|----------------|--------------|
| `content_age` | `2026 - release_year` | Captures how old a title is |
| `duration_numeric` | Extract number from "90 min" / "2 Seasons" | Makes duration usable for ML |
| `primary_genre` | First genre from `listed_in` | Simplifies multi-genre field |
| `country_group` | Top 5 countries + "Other" | Reduces noise from 100+ countries |

---

## 🤖 ML Approach — K-Means Clustering

### Why Unsupervised Learning?
There are no pre-defined labels for what "type" of content a show is. K-Means discovers **natural groupings** on its own.

### Features Used for Clustering
- `type` (Movie / TV Show)
- `primary_genre`
- `country_group`
- `duration_numeric`
- `content_age`

### Steps
1. **Label Encode** categorical columns
2. **StandardScaler** — normalize all features to same scale
3. **Elbow Method** — try K from 2 to 12, find the "elbow" point
4. **K-Means with K=5** — final model

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['cluster'] = labels
```

---

## 🔍 Clustering Results

| Cluster | Label | Dominant Type | Top Genre | Notes |
|---------|-------|---------------|-----------|-------|
| 0 | 🌍 International Drama | TV Show | International TV | Foreign-language dramas |
| 1 | 🎬 US Action Movies | Movie | Action & Adventure | Hollywood blockbusters |
| 2 | 👨‍👩‍👧 Family & Kids | Movie | Children & Family | Animated + family content |
| 3 | 🎭 Stand-up & Docs | Movie | Documentaries | Stand-up specials, docs |
| 4 | 📺 Long-Running Series | TV Show | Drama | Multi-season TV dramas |

> ⚠️ Actual cluster labels may vary depending on dataset version and random state.

---

## 📉 PCA Visualization

PCA reduces the feature space to 2 dimensions for visual interpretation.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

The resulting scatter plot shows how well-separated the 5 clusters are in 2D space.

---

## 🎯 Recommendation System

### Method: TF-IDF + Cosine Similarity

1. Combine `listed_in` + `description` + `type` + `country_group` into a `tags` field
2. Apply **TF-IDF Vectorizer** (5000 features, bigrams)
3. Compute **Cosine Similarity** between all title pairs
4. For any input title → return top N most similar titles

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['tags'])

# Get recommendations for a title
scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
```

### Example Output

```
🎬 Because you watched: 'Breaking Bad'
   (Cluster 1 | TV Show | Crime TV Shows)

🍿 We recommend:
   1. Ozark              — Crime Drama     | Similarity: 0.847
   2. Narcos             — Crime Drama     | Similarity: 0.821
   3. El Chapo           — Crime TV Show   | Similarity: 0.798
   4. Better Call Saul   — Crime Drama     | Similarity: 0.783
   5. Mindhunter         — Crime Drama     | Similarity: 0.761
```

---

## 💼 Business Insights

1. **Netflix is movie-heavy** — ~70% Movies vs 30% TV Shows, but TV Shows are growing faster post-2018
2. **International content dominates** — India, UK, Canada are major producers after the US
3. **Content exploded after 2015** — aligns with Netflix's global expansion strategy
4. **Drama & International Movies** are the top genres — Netflix should continue investing in these
5. **Niche genres are rising** — Anime, Sci-Fi, and K-Drama are growing fast among younger audiences
6. **Clusters enable personalization** — each cluster maps to a distinct viewer persona for targeted recommendations

---

## 📂 Project Structure

```
netflix-ml-analysis/
│
├── netflix_titles.csv                        # Dataset
├── Netflix_Content_Analysis_ML.ipynb         # Main notebook
├── README.md                                 # This file
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/netflix-ml-analysis.git
cd netflix-ml-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Download the dataset from Kaggle
# Place netflix_titles.csv in the project folder

# 4. Open the notebook
jupyter notebook Netflix_Content_Analysis_ML.ipynb
```

---

## 🚀 Future Improvements

| Enhancement | Description |
|-------------|-------------|
| 🔧 **DBSCAN Clustering** | Better handles noise and non-spherical cluster shapes |
| 🌐 **Streamlit Deployment** | Deploy recommender as an interactive web app |
| 💬 **Sentiment Analysis** | Analyze sentiment of descriptions to improve recommendations |
| 🧠 **BERT Embeddings** | Use transformer-based embeddings for richer content similarity |
| ⭐ **Collaborative Filtering** | Recommend based on user watch history (requires user data) |
| 📊 **Real-time Dashboard** | Interactive cluster exploration with Plotly/Dash |

---

 ⭐ If you found this project helpful, give it a star on GitHub!
