# MovieLens Recommender: Addressing User Cold-Start Recommendation with LLM Data Augmentation

Make sure to download and extract the datasets into the `data` directory.

Links to datasets:
https://grouplens.org/datasets/movielens/100k/ 
https://grouplens.org/datasets/movielens/1m/


This project explores the use of **Large Language Models (LLMs)** to address the **cold-start problem** in recommendation systems (RecSys). Traditional RecSys methods struggle to provide meaningful recommendations for new users with limited interaction history. We leverage **LLaMA-3B** to generate **synthetic user-item interactions**, improving recommendations for cold-start users.

## How It Works
1. **Data Processing**: We use the **MovieLens 100K and 1M datasets**, mapping user and item IDs to structured numerical formats.
2. **Cold-Start Simulation**: We artificially introduce cold users by restricting their interaction history to a single item.
3. **LLM Data Augmentation**: We employ **pairwise prompting** with **LLaMA-3B** to simulate user preferences, enriching training data.
4. **Model Training**: We train **Two-Tower Matrix Factorization (MF)** and **SASRec** (Self-Attentive Sequential Recommendation) models using both original and augmented datasets.
5. **Evaluation**: We assess **Hit Rate (HR@K)** and **Normalized Discounted Cumulative Gain (NDCG@K)** to measure recommendation quality.

## Expected Results
- **Enhanced Recommendations**: LLM-augmented interactions aim to improve RecSys performance, particularly for cold users.
- **Comparative Analysis**: We evaluate whether **LLM-based augmentation** outperforms traditional cold-start approaches.
- **Optimized Prompting**: Future work includes refining **in-context learning** to enhance LLM-driven recommendations.

📌 Authors: Nachiket Subbaraman, Georgy Zaets, Anant Vishwakarma, Zeerak Babar, Jaskinder Sarai
📧 Contact: [nsubbaraman, gzaets, abvishwakarma, zebabar, jssarai]@ucdavis.edu

Let us know if you want any refinements! 🚀
