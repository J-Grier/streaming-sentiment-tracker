# Streaming Sentiment Tracker 🎬  
**NLP Sentiment Analysis of Fan Engagement Across Streaming Platforms**

This project leverages advanced natural language processing (NLP) techniques to predict audience sentiment and overall reception of streaming shows. By analyzing social media and streaming platforms, specifically Reddit and YouTube, the tracker delivers insights on audience engagement and reception, providing strategic recommendations for content creators and marketers.The goal is to understand how hype builds, which platforms drive the most engagement, and whether early sentiment predicts future success.

---

## Objectives

**Primary Goal**: To accurately forecast the public reception of streaming TV series using sentiment analysis and topic modeling.

**Actionable Insights**: Provide strategic insights that inform content development, marketing strategies, and audience engagement.

---
## Approach

Several NLP methods were integrated to effectively capture and analyze sentiment:

**Logistic Regression and Random Forest Classifiers**: Established baseline models using TF-IDF vectorization, highlighting initial challenges with nuanced sentiment.

-**Sentiment Analysis (Bidirectional LSTM)**: Implemented a deep learning model with trainable embeddings (FastText and Emoji2Vec), significantly outperforming traditional methods (accuracy improved from ~82.3% baseline to 85.0%). Notably, the LSTM model consistently outperformed VADER, providing greater accuracy on human-labeled datasets.

-**Topic Modeling (Latent Dirichlet Allocation)**: Applied to identify key discussion themes and context, allowing us to link specific sentiment reactions to clear topical clusters.

By pairing:

-**LSTM** for text-level precision,

-**LDA** to understand topic context,

the project created a multi-layered sentiment engine that clearly defines audience sentiment and contextualizes their reactions effectively.

## Key Findings

-**Advanced Sentiment Prediction**: The Bidirectional LSTM provided superior classification accuracy, significantly outperforming baseline models and traditional VADER approaches.

-**Distinct Audience Segments**: Clearly identified audience reactions and thematic differences, enabling targeted marketing and content strategies.-

-**Contextual Understanding through Topic Modeling**: LDA revealed specific topics driving both positive and negative audience engagement, enriching sentiment insights.

## Impact and Applications

-**Strategic Content Decisions**: Empowers content creators and marketers to proactively tailor content and campaigns based on predicted audience reception.

-**Enhanced Audience Targeting**: Clarifies viewer preferences, enabling more effective targeting and engagement strategies.

-**Risk Mitigation**: Offers predictive insights to proactively address potential negative audience reactions.

## Limitations and Future Directions

-**Platform Limitations**: Reddit and YouTube data may miss sentiments expressed on other platforms (e.g., Twitter, Instagram).

-**Potential Model Enhancements**: Integrating more diverse data sources and leveraging additional deep learning architectures could further enhance predictive accuracy and context sensitivity.

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
