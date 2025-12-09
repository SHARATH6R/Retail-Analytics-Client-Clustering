Retail Analytics – Client Clustering

This project segments e-commerce customers using RFM scoring and K-Means clustering to help businesses identify high-value buyers, slipping customers, and growth opportunities.

What’s Inside
Cleaned retail transaction data
RFM feature generation
PCA for dimensionality reduction
K-Means clustering
Simple profiles of each customer group

Key Insights
A small group drives most revenue
Three clear segments: Premium, Standard, At-Risk
Strong seasonal patterns
UK customers dominate high-value purchases

Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, NLTK

Project Structure
analysis_notebook.ipynb
client_segmentation_results.csv
data/
output_graphs/
visualization_assets/

Run
pip install -r requirements.txt
jupyter notebook analysis_notebook.ipynb

import nltk
nltk.download('stopwords')
nltk.download('punkt')
