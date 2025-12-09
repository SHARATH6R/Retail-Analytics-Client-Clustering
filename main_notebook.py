import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

transaction_df = pd.read_excel('transaction_records.xlsx', sheet_name='Online Retail')

print(f'Dataset dimensions: {transaction_df.shape}')
print(f'\nColumn specifications:')
print(transaction_df.dtypes)
print(f'\nInitial rows preview:')
transaction_df.head()

print('\nNull value assessment:')
print(transaction_df.isnull().sum())
print(f'\nNull percentage:')
print(round(transaction_df.isnull().sum() / len(transaction_df) * 100, 2))

print('\nDuplicate records:', transaction_df.duplicated().sum())
print('\nBasic statistics:')
transaction_df.describe()

cleaned_data = transaction_df.dropna(subset=['ClientIdentifier'])
cleaned_data = cleaned_data[cleaned_data['ClientIdentifier'].notna()]

print(f'Records after null removal: {len(cleaned_data)}')
print(f'Records dropped: {len(transaction_df) - len(cleaned_data)}')

cleaned_data = cleaned_data[cleaned_data['TransactionReference'].str.contains('C') == False]
cleaned_data = cleaned_data[cleaned_data['PricePerUnit'] > 0]
cleaned_data = cleaned_data[cleaned_data['OrderQuantity'] > 0]

print(f'Records after filtering negatives: {len(cleaned_data)}')

cleaned_data = cleaned_data.drop_duplicates()
print(f'Records after duplicate removal: {len(cleaned_data)}')

def identify_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_value = q3 - q1
    lower_fence = q1 - (1.5 * iqr_value)
    upper_fence = q3 + (1.5 * iqr_value)
    return (series < lower_fence) | (series > upper_fence)

quantity_outliers = identify_outliers_iqr(cleaned_data['OrderQuantity'])
price_outliers = identify_outliers_iqr(cleaned_data['PricePerUnit'])

print(f'Quantity outliers: {quantity_outliers.sum()}')
print(f'Price outliers: {price_outliers.sum()}')

cleaned_data = cleaned_data[~(quantity_outliers | price_outliers)]
print(f'Records after outlier removal: {len(cleaned_data)}')

cleaned_data['TransactionValue'] = cleaned_data['OrderQuantity'] * cleaned_data['PricePerUnit']
cleaned_data['TransactionDate'] = pd.to_datetime(cleaned_data['TransactionDate'])
cleaned_data['TransactionYear'] = cleaned_data['TransactionDate'].dt.year
cleaned_data['TransactionMonth'] = cleaned_data['TransactionDate'].dt.month
cleaned_data['TransactionDay'] = cleaned_data['TransactionDate'].dt.day
cleaned_data['TransactionHour'] = cleaned_data['TransactionDate'].dt.hour
cleaned_data['TransactionDayName'] = cleaned_data['TransactionDate'].dt.day_name()

print('Feature engineering completed')
print(cleaned_data.head())

analysis_reference_date = cleaned_data['TransactionDate'].max() + timedelta(days=1)
print(f'Reference date for recency calculation: {analysis_reference_date}')

client_aggregation = cleaned_data.groupby('ClientIdentifier').agg({
    'TransactionDate': lambda x: (analysis_reference_date - x.max()).days,
    'TransactionReference': 'nunique',
    'TransactionValue': 'sum'
}).reset_index()

client_aggregation.columns = ['ClientIdentifier', 'DaysSinceLastOrder', 'TransactionCount', 'TotalSpending']

print('RFM metrics computed:')
print(client_aggregation.head())
print(f'\nRFM statistics:')
print(client_aggregation.describe())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(client_aggregation['DaysSinceLastOrder'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Distribution of Recency', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Days Since Last Order')
axes[0].set_ylabel('Client Count')

axes[1].hist(client_aggregation['TransactionCount'], bins=50, color='coral', edgecolor='black')
axes[1].set_title('Distribution of Frequency', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Transaction Count')
axes[1].set_ylabel('Client Count')

axes[2].hist(client_aggregation['TotalSpending'], bins=50, color='mediumseagreen', edgecolor='black')
axes[2].set_title('Distribution of Monetary Value', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Total Spending')
axes[2].set_ylabel('Client Count')

plt.tight_layout()
plt.show()

def calculate_rfm_score(dataframe, metric_name, quartiles=4, ascending=True):
    return pd.qcut(dataframe[metric_name], q=quartiles, labels=range(1, quartiles + 1), duplicates='drop')

client_aggregation['RecencyScore'] = calculate_rfm_score(client_aggregation, 'DaysSinceLastOrder', ascending=False)
client_aggregation['FrequencyScore'] = calculate_rfm_score(client_aggregation, 'TransactionCount', ascending=True)
client_aggregation['MonetaryScore'] = calculate_rfm_score(client_aggregation, 'TotalSpending', ascending=True)

client_aggregation['RecencyScore'] = client_aggregation['RecencyScore'].astype(int)
client_aggregation['FrequencyScore'] = client_aggregation['FrequencyScore'].astype(int)
client_aggregation['MonetaryScore'] = client_aggregation['MonetaryScore'].astype(int)

client_aggregation['ComprehensiveRFM'] = (
    client_aggregation['RecencyScore'].astype(str) +
    client_aggregation['FrequencyScore'].astype(str) +
    client_aggregation['MonetaryScore'].astype(str)
)

client_aggregation['AggregateScore'] = (
    client_aggregation['RecencyScore'] +
    client_aggregation['FrequencyScore'] +
    client_aggregation['MonetaryScore']
)

print('RFM scoring completed:')
print(client_aggregation.head(10))

segment_classification = {
    'Elite': (client_aggregation['AggregateScore'] >= 9),
    'Loyal': ((client_aggregation['AggregateScore'] >= 6) & (client_aggregation['AggregateScore'] < 9)),
    'Potential': ((client_aggregation['AggregateScore'] >= 4) & (client_aggregation['AggregateScore'] < 6)),
    'AtRisk': (client_aggregation['AggregateScore'] < 4)
}

for segment_label, condition in segment_classification.items():
    client_aggregation.loc[condition, 'ClientSegment'] = segment_label

segment_distribution = client_aggregation['ClientSegment'].value_counts()
print('\nSegment distribution:')
print(segment_distribution)

plt.figure(figsize=(10, 6))
segment_distribution.plot(kind='bar', color=['gold', 'limegreen', 'dodgerblue', 'tomato'], edgecolor='black')
plt.title('Client Segment Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Segment Category')
plt.ylabel('Client Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

segment_profiles = client_aggregation.groupby('ClientSegment').agg({
    'DaysSinceLastOrder': 'mean',
    'TransactionCount': 'mean',
    'TotalSpending': 'mean'
}).round(2)

print('\nSegment behavioral profiles:')
print(segment_profiles)

product_description_df = cleaned_data[['ProductSKU', 'ItemDescription']].drop_duplicates()
print(f'Unique products: {len(product_description_df)}')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text_content):
    if pd.isna(text_content):
        return ''
    text_content = text_content.lower()
    text_content = re.sub(r'[^a-zA-Z\s]', '', text_content)
    tokens = word_tokenize(text_content)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_tokens)

product_description_df['ProcessedDescription'] = product_description_df['ItemDescription'].apply(preprocess_text)

tfidf_transformer = TfidfVectorizer(max_features=100)
product_features = tfidf_transformer.fit_transform(product_description_df['ProcessedDescription'])

print(f'TF-IDF matrix shape: {product_features.shape}')

inertia_values = []
silhouette_values = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_model.fit(product_features)
    inertia_values.append(kmeans_model.inertia_)
    silhouette_values.append(silhouette_score(product_features, kmeans_model.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cluster_range, inertia_values, marker='o', color='darkblue', linewidth=2)
axes[0].set_title('Elbow Method for Product Clustering', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

axes[1].plot(cluster_range, silhouette_values, marker='s', color='darkred', linewidth=2)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()

optimal_product_clusters = 5
final_kmeans = KMeans(n_clusters=optimal_product_clusters, random_state=42, n_init=10)
product_description_df['ProductCategory'] = final_kmeans.fit_predict(product_features)

print(f'\nProduct category distribution:')
print(product_description_df['ProductCategory'].value_counts())

cleaned_data = cleaned_data.merge(
    product_description_df[['ProductSKU', 'ProductCategory']],
    on='ProductSKU',
    how='left'
)

category_spending = cleaned_data.groupby(['ClientIdentifier', 'ProductCategory'])['TransactionValue'].sum().unstack(fill_value=0)
category_spending.columns = [f'Category_{int(col)}_Spending' for col in category_spending.columns]

client_aggregation = client_aggregation.merge(category_spending, on='ClientIdentifier', how='left')
client_aggregation = client_aggregation.fillna(0)

print('Enhanced client features:')
print(client_aggregation.head())

rfm_features = client_aggregation[['DaysSinceLastOrder', 'TransactionCount', 'TotalSpending']]

print('\nApplying log transformation to reduce skewness:')
rfm_log_transformed = np.log1p(rfm_features)

feature_scaler = StandardScaler()
rfm_scaled = feature_scaler.fit_transform(rfm_log_transformed)

print('Feature scaling completed')
print(f'Scaled feature shape: {rfm_scaled.shape}')

pca_model = PCA(n_components=0.95)
pca_transformed = pca_model.fit_transform(rfm_scaled)

print(f'\nPCA components retained: {pca_model.n_components_}')
print(f'Explained variance ratio: {pca_model.explained_variance_ratio_}')
print(f'Cumulative explained variance: {pca_model.explained_variance_ratio_.sum():.4f}')

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_model.explained_variance_ratio_) + 1), 
         np.cumsum(pca_model.explained_variance_ratio_), 
         marker='o', linestyle='--', color='purple', linewidth=2)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% Variance Threshold')
plt.title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

inertia_clustering = []
silhouette_clustering = []
cluster_test_range = range(2, 11)

for k in cluster_test_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_transformed)
    inertia_clustering.append(kmeans.inertia_)
    silhouette_clustering.append(silhouette_score(pca_transformed, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cluster_test_range, inertia_clustering, marker='o', color='navy', linewidth=2)
axes[0].set_title('Elbow Method for Client Clustering', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

axes[1].plot(cluster_test_range, silhouette_clustering, marker='s', color='darkgreen', linewidth=2)
axes[1].set_title('Silhouette Analysis for Client Clustering', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()

optimal_client_clusters = 3
final_client_kmeans = KMeans(n_clusters=optimal_client_clusters, random_state=42, n_init=10)
client_aggregation['ClusterAssignment'] = final_client_kmeans.fit_predict(pca_transformed)

print(f'\nCluster distribution:')
print(client_aggregation['ClusterAssignment'].value_counts())

cluster_characteristics = client_aggregation.groupby('ClusterAssignment').agg({
    'DaysSinceLastOrder': ['mean', 'median'],
    'TransactionCount': ['mean', 'median'],
    'TotalSpending': ['mean', 'median']
}).round(2)

print('\nCluster behavioral characteristics:')
print(cluster_characteristics)

client_aggregation['PrimaryComponent'] = pca_transformed[:, 0]
client_aggregation['SecondaryComponent'] = pca_transformed[:, 1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    client_aggregation['PrimaryComponent'],
    client_aggregation['SecondaryComponent'],
    c=client_aggregation['ClusterAssignment'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='black',
    s=50
)
plt.title('Client Clusters in PCA Space', fontsize=16, fontweight='bold')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster Assignment')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig = px.scatter_3d(
    client_aggregation,
    x='DaysSinceLastOrder',
    y='TransactionCount',
    z='TotalSpending',
    color='ClusterAssignment',
    title='3D Visualization of Client Clusters',
    labels={
        'DaysSinceLastOrder': 'Recency (Days)',
        'TransactionCount': 'Frequency',
        'TotalSpending': 'Monetary Value'
    },
    color_continuous_scale='Viridis'
)
fig.show()

cluster_labels = {
    0: 'High Value Clients',
    1: 'Medium Value Clients',
    2: 'Low Value Clients'
}

cluster_summary = client_aggregation.groupby('ClusterAssignment').agg({
    'ClientIdentifier': 'count',
    'DaysSinceLastOrder': 'mean',
    'TransactionCount': 'mean',
    'TotalSpending': 'mean'
}).round(2)

cluster_summary.columns = ['ClientCount', 'AvgRecency', 'AvgFrequency', 'AvgMonetary']
cluster_summary['ClusterLabel'] = cluster_summary.index.map(cluster_labels)

print('\nFinal cluster summary:')
print(cluster_summary)

client_aggregation.to_csv('client_segmentation_results.csv', index=False)
print('\nAnalysis complete. Results saved to client_segmentation_results.csv')
