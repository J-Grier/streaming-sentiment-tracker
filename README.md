# Streaming Sentiment Tracker 🎬  
**NLP Sentiment Analysis of Trailer Hype Across Streaming Platforms**

This project analyzes public sentiment before and after major promotional events (like trailer drops or Netflix’s TUDUM) for major streaming shows. The goal is to understand how hype builds, which platforms drive the most engagement, and whether early sentiment predicts future success.

---

## 📁 Project Structure

```plaintext
project/
│
├── data/
│   ├── raw/              # Raw scraped data (e.g., ahsoka_youtube_comments.csv)
│   └── processed/        # Cleaned and labeled data ready for modeling
│
├── notebooks/
│   ├── ScrapingandSentiment.ipynb    # YouTube + Reddit scraping, cleaning + VADER labeling
│   └── SSSmodel.ipynb                # Sentiment modeling (LogReg, RF, LSTM), LDA, and show-level forecasting
│
├── utils/
│   └── cleaning.py       # Custom text preprocessing functions
│
└── README.md             # Project overview and structure

---

## 🧠 Final Model: Predicting Reception from Fan Discourse

The final stage of this project builds a proof-of-concept model to predict whether a show will succeed (based on Rotten Tomatoes audience score) using only early fan discussion.

Each show is represented by:
- ✅ **LSTM-predicted sentiment** (Reddit + YouTube)
- ✅ **LDA-based topic proportions** from Reddit
- ✅ 🧮 **Comment volume**

A simple logistic regression model trained on *Velma* and *The Bear S3* correctly classified both and was then applied to *The Last of Us Season 2* as a predictive test case — where it confidently predicted success.

> While not statistically powerful with only two labeled examples, this model illustrates how tone + theme in fan conversation can forecast audience reception.

**Future work** could include integrating VADER to provide an additional layer of emotional tone intensity.

---
