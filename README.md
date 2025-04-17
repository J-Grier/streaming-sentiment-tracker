# Streaming Sentiment Tracker 🎬  
**NLP Sentiment Analysis of Trailer Hype Across Streaming Platforms**

This project analyzes public sentiment before and after major promotional events (like trailer drops or Netflix’s TUDUM) for major streaming shows. The goal is to understand how hype builds, which platforms drive the most engagement, and whether early sentiment predicts future success.

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

## 📁 Project Structure

```plaintext
streaming-sentiment-tracker/
│
├── Notebooks/
│   ├── SSSmodel.ipynb              # Sentiment modeling, topic modeling, and forecasting
│   └── ScrapingandSentiment.ipynb  # (Uploaded via GitHub Web) Reddit/YouTube scraping + LSTM labeling
│
├── Utils/
│   └── cleaning.py                 # Custom text preprocessing functions for sentiment & LDA
│
├── README.md                       # Project overview and results summary
├── .gitignore                      # Excludes models, large assets, and raw data
