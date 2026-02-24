# 🧠 Mental Health Sentiment Monitor
### Social Media Big Data Analytics for Public Sentiment Monitoring

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.0.2-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete Big Data NLP pipeline for analyzing public sentiment across Reddit mental health communities. Combines rule-based, statistical, and deep learning models to extract sentiment, emotions, and topics from social media data — with real-time monitoring via the HackerNews Live API and distributed processing of 1.6 million tweets using Apache Spark.

---

## 📸 Dashboard Preview

| Overview & Sentiment | Emotion Analysis | Topic Modeling |
|---|---|---|
| 3-model sentiment comparison | Emotion × Community heatmap | BERTopic topic discovery |
| Model disagreement analysis | Treemap & stacked bars | Sunburst & bubble charts |
| Live metrics | 6-emotion classification | 4 discovered topics |

---

## 🏗️ Architecture

```
Reddit Mental Health Data (4,620 posts)
        │
        ▼
Preprocessing Pipeline
(cleaning, stopwords, lemmatization)
        │
        ├──── VADER (Rule-based sentiment)
        ├──── Logistic Regression (TF-IDF + Scikit-learn)
        ├──── DistilBERT (Transformer sentiment)
        ├──── Emotion Classifier (6 emotions)
        └──── BERTopic (Topic modeling)
        │
        ├──── PySpark MLlib (4,620 posts)
        ├──── PySpark Big Data (1,600,000 tweets — Sentiment140)
        └──── HackerNews Live API (200 real-time posts)
        │
        ▼
Streamlit Dashboard (11 sections)
```

---

## 📊 Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Reddit Mental Health | Kaggle | 4,620 posts | Domain-specific NLP analysis |
| Sentiment140 | Kaggle | 1,600,000 tweets | Big data scale validation |
| HackerNews Live | Public API | 200 posts (live) | Real-time monitoring |

**Reddit Communities covered:**
- r/Depression
- r/Anxiety
- r/MentalHealth
- r/SocialAnxiety
- r/Mindfulness

---

## 🤖 Models Used

| Model | Type | Purpose | Accuracy |
|---|---|---|---|
| VADER | Rule-based lexicon | Fast sentiment scoring | Baseline |
| Logistic Regression (Scikit-learn) | Statistical ML | TF-IDF sentiment classification | 75.93% |
| DistilBERT | Transformer (Deep Learning) | Context-aware sentiment | Best on nuanced text |
| Emotion Classifier | Transformer | 6-emotion classification | Pre-trained |
| BERTopic | Clustering + NLP | Unsupervised topic discovery | State-of-the-art |
| Spark MLlib LR | Distributed ML | Scalable classification | 86.68% |

---

## 🔑 Key Findings

1. **DistilBERT vs VADER disagreement** — Posts like *"I'm desperate for a friend"* are scored **positive** by VADER (it detected "friend") but **negative** by DistilBERT (it understood desperation). This proves context-aware models are essential for mental health text.

2. **Depression is 92.6% negative** — The highest negativity rate of all five communities, validated independently by both Python and Spark pipelines.

3. **Mindfulness community's dominant emotion is Fear (35.7%)** — Not joy. People seek mindfulness *because* they're anxious, not after finding peace. This is a counterintuitive finding from the data.

4. **Spark MLlib outperforms scikit-learn** — 86.68% vs 75.93% accuracy using distributed TF-IDF with 10,000 features.

5. **Big Data at scale** — Apache Spark 4.0 loaded 1,600,000 tweets in 7.4 seconds and trained on 1,280,209 samples in 29.7 seconds on a MacBook Air M3.

---

## 🗂️ Project Structure

```
mental_health_sentiment/
│
├── data/
│   ├── reddit_mental_health.csv      ← Raw Reddit dataset
│   ├── sentiment140.csv              ← Sentiment140 (1.6M tweets)
│   ├── processed_data.csv            ← After preprocessing
│   ├── sentiment_results.csv         ← After 3-model scoring
│   ├── emotion_results.csv           ← After emotion classification
│   ├── final_results.csv             ← After topic modeling
│   └── live_stream.csv               ← HackerNews live posts
│
├── models/
│   ├── lr_metrics.json               ← Logistic Regression metrics
│   ├── spark_metrics.json            ← Spark MLlib metrics
│   ├── bigdata_metrics.json          ← Sentiment140 Spark metrics
│   ├── stream_summary.json           ← Live stream summary
│   ├── topic_info.csv                ← BERTopic topic info
│   └── topic_words.json              ← Topic keywords
│
├── 2_preprocess.py                   ← Data cleaning pipeline
├── 3_sentiment_models.py             ← VADER + LR + DistilBERT
├── 4_emotion_classifier.py           ← 6-emotion classification
├── 5_topic_modeling.py               ← BERTopic topic discovery
├── 6_dashboard.py                    ← Streamlit dashboard
├── 7_spark_analysis.py               ← PySpark on Reddit data
├── 8_spark_bigdata.py                ← PySpark on 1.6M tweets
├── 9_realtime_stream.py              ← HackerNews live stream
├── explore.py                        ← Data exploration script
├── config.py                         ← Configuration file
├── requirements.txt                  ← Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9
- Java 17 (required for Apache Spark)
- Mac M3 / any modern system with 8GB+ RAM

### Step 1 — Install Java 17
```bash
brew install openjdk@17
echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@17' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
```

### Step 2 — Clone the repository
```bash
git clone https://github.com/yourusername/mental_health_sentiment.git
cd mental_health_sentiment
```

### Step 3 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Download datasets
- **Reddit Mental Health:** [Kaggle — Reddit Mental Health Data](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data) → save as `data/reddit_mental_health.csv`
- **Sentiment140:** [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) → save as `data/sentiment140.csv`

---

## 🚀 Running the Pipeline

Run scripts in order. Each builds on the previous output.

```bash
# Activate environment first
source venv/bin/activate
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH=$JAVA_HOME/bin:$PATH

# Step 1 — Preprocess Reddit data
python3 2_preprocess.py

# Step 2 — Run 3-model sentiment analysis
python3 3_sentiment_models.py

# Step 3 — Run emotion classification
python3 4_emotion_classifier.py

# Step 4 — Run topic modeling
python3 5_topic_modeling.py

# Step 5 — PySpark analysis on Reddit data
python3 7_spark_analysis.py

# Step 6 — PySpark on 1.6M Sentiment140 tweets
python3 8_spark_bigdata.py

# Step 7 — Collect HackerNews live stream
python3 9_realtime_stream.py

# Step 8 — Launch dashboard
python3 -m streamlit run 6_dashboard.py
```

Dashboard opens at **http://localhost:8501**

---

## 📈 Dashboard Sections

| Section | Description |
|---|---|
| Overview | Key metrics — total posts, sentiment %, dominant emotion, model agreement |
| Sentiment Distribution | VADER vs DistilBERT donut charts + 3-model bar comparison |
| Model Performance | Model comparison, accuracy, disagreement examples |
| Emotion Classification | Treemap, heatmap, stacked community breakdown |
| Topic Modeling | BERTopic topics with sunburst and bubble charts |
| Word Clouds | Most common words in negative vs positive posts |
| Community Deep Dive | Per-community negativity rates and stats |
| Data Explorer | Filterable raw data table |
| PySpark Analytics | Spark SQL results + MLlib accuracy |
| Big Data Scale | Sentiment140 1.6M tweet processing results |
| Live Stream | Real-time HackerNews sentiment monitoring |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9 |
| Big Data | Apache Spark 4.0, PySpark, Spark MLlib |
| NLP Models | HuggingFace Transformers, DistilBERT, BERTopic |
| ML | Scikit-learn, TF-IDF, Logistic Regression |
| Sentiment | VADER, DistilBERT SST-2 |
| Emotion | distilbert-base-uncased-emotion |
| Topic Modeling | BERTopic, Sentence Transformers |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, WordCloud |
| Data | Pandas, NumPy |
| Live API | HackerNews Firebase API |

---

## 📝 Results Summary

| Metric | Value |
|---|---|
| Total Reddit posts analyzed | 4,620 |
| Sentiment140 tweets processed | 1,600,000 |
| Live posts streamed | 200 |
| Spark data load time (1.6M records) | 7.4 seconds |
| Spark training time (1.28M samples) | 29.7 seconds |
| Spark MLlib accuracy | 86.68% |
| Scikit-learn LR accuracy | 75.93% |
| Model agreement rate | 65% |
| Topics discovered | 4 |
| Emotions classified | 6 |
| Communities analyzed | 5 |

---

## 👤 Author

**Abhishek Reddy Kotha**
Academic Project — Social Media Big Data Analytics for Public Sentiment Monitoring

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) — Go et al., Stanford University
- [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data) — Kaggle
- [BERTopic](https://maartengr.github.io/BERTopic/) — Maarten Grootendorst
- [HuggingFace Transformers](https://huggingface.co/) — for DistilBERT and emotion models
- [HackerNews API](https://github.com/HackerNews/API) — Y Combinator
