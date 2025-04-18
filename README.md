# 🎬 Streaming Sentiment Tracker: Predicting Audience Reception

## 📈 Overview
This project leverages advanced natural language processing (NLP) and deep learning techniques to analyze public sentiment toward streaming TV series. By processing user comments from YouTube and Reddit, the model predicts audience reception and provides actionable insights into both audience sentiment (positive, negative, neutral) and the underlying themes driving public discussion.

---

## 🎯 Objectives
- **Primary Goal**: Predict and analyze audience sentiment toward streaming shows.
- **Actionable Insights**: Provide strategic insights for content creators and marketers based on predicted audience reactions and thematic analysis.

---

## 🛠️ Methodology

The analysis employed a layered modeling approach combining sentiment analysis and topic modeling:

### Data Collection
- Collected YouTube and Reddit comments covering released, ongoing, and upcoming streaming series.
- Focused collection windows tailored to show release schedules.

### Sentiment Analysis (VADER & LSTM)
- **VADER** provided baseline sentiment scoring (positive, neutral, negative) suitable for aggregate-level insights.
- **LSTM (Bidirectional)** improved context-aware sentiment predictions, capturing nuances missed by simpler models (e.g., sarcasm, negation, multi-part sentiment).

### Topic Modeling (LDA)
- Extracted interpretable themes from discussions, revealing the underlying topics driving sentiment reactions.

---

## 📊 Model Pipeline Summary
1. **Preprocessing**
   - Cleaned and tokenized Reddit and YouTube comments.
   - Generated sentiment scores using VADER and LSTM models.
   - Computed topic distributions using LDA.
   
2. **Feature Engineering**
   - Created comprehensive feature sets from sentiment proportions and topic distributions.

3. **Final Reception Prediction**
   - Used Logistic Regression to predict show success based on Rotten Tomatoes audience scores.

---

## 🔑 Key Findings

### 🚀 Sentiment Model Performance
| Model                            | Accuracy      | Strengths                                   | Weaknesses                   |
|----------------------------------|---------------|---------------------------------------------|------------------------------|
| Logistic Regression (TF-IDF)     | ~82.3%        | Strong baseline                             | Struggles with context       |
| Random Forest                    | ~76.8%        | Captures nonlinearities                     | Overfits nuances             |
| Vanilla LSTM                     | ~83.5%        | Improved context handling                   | Neutral sentiment precision  |
| **Bidirectional LSTM (Final)**   | **85.0%** 🚀  | Best overall; nuanced context capture       | Slightly underpredicts neutral |

### 🧪 Topic Modeling Insights (LDA)
- Identified specific topics driving audience sentiment (e.g., character arcs, social issues).
- Clarified reasons behind positive or negative sentiment for specific shows like *Velma* and *The Bear*.

### 🎯 Final Reception Model
- Accurately predicted *Velma* as a flop and *The Bear Season 3* as a success, achieving ~91% confidence.
- Successfully generalized to predict *The Last of Us Season 2* as a success with high confidence.

---
## 🧠 Why Bidirectional LSTM?

Bidirectional LSTMs process text forwards and backwards, effectively capturing nuanced sentiments often missed by simpler models:

![Bidirectional LSTM Equations](assets/lstm_equations.png)

This mathematical structure allows the model to dynamically interpret complex sentiments, making it particularly effective for analyzing nuanced and long-form user-generated content.

## ✅ Conclusion and Practical Impact
By combining sentiment analysis (VADER & LSTM) and topic modeling (LDA), this project demonstrates how nuanced analysis of online discourse can accurately forecast audience reception. This layered sentiment engine provides actionable insights for content creators, marketers, and platforms seeking to anticipate and respond to audience sentiment proactively.

## 🚧 Limitations and Future Directions

### Limitations:
-Short-form and sarcastic comments present challenges.
-Model currently trained on limited labeled examples, limiting generalizability.

### Future Work:

| Idea                           | Goal                                              |
|--------------------------------|---------------------------------------------------|
| Add VADER as additional layer  | Enhance emotional tone calibration                |
| Incorporate YouTube extensively| Better sentiment detection for polarizing content |
| Extend labeled datasets        | Improve model generalization                      |
| Include time-series modeling   | Capture sentiment shifts over episode releases    |

## 🔖 References

- Hochreiter, S., & Schmidhuber, J. (1997). [*Long Short-Term Memory*](https://pdfs.semanticscholar.org/0027/d572e43d0c120d59e81c228f2a17b3b05006.pdf). *Neural Computation, 9*(8), 1735–1780.
  - This paper introduces Long Short-Term Memory (LSTM) networks, which effectively model sequential data and overcome the vanishing gradient problem common in traditional recurrent neural networks (RNNs), making them highly suitable for NLP and sentiment analysis tasks.
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [LDA Topic Modeling](https://radimrehurek.com/gensim/models/ldamodel.html)
- [Reddit](https://reddit.com)
- [YouTube](https://youtube.com)

---
## Author

**John Grier** 
MS Data Science Candidate, Illinois Tech
GitHub: J-Grier

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
```
