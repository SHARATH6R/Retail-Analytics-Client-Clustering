import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class DataQualityAnalyzer:
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.original_shape = dataframe.shape
        
    def assess_missing_values(self) -> pd.DataFrame:
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        
        assessment_df = pd.DataFrame({
            'column': null_counts.index,
            'null_count': null_counts.values,
            'null_percentage': null_percentages.values
        })
        
        return assessment_df[assessment_df['null_count'] > 0].sort_values('null_count', ascending=False)
    
    def identify_duplicates(self) -> Dict[str, int]:
        duplicate_count = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(self.df)) * 100
        
        return {
            'duplicate_records': duplicate_count,
            'duplicate_percentage': round(duplicate_percentage, 2)
        }
    
    def detect_outliers_iqr(self, column: str) -> pd.Series:
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr_range = q3 - q1
        
        lower_boundary = q1 - (1.5 * iqr_range)
        upper_boundary = q3 + (1.5 * iqr_range)
        
        outlier_mask = (self.df[column] < lower_boundary) | (self.df[column] > upper_boundary)
        return outlier_mask
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> pd.Series:
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outlier_indices = z_scores > threshold
        
        result_mask = pd.Series(False, index=self.df.index)
        result_mask.loc[self.df[column].notna()] = outlier_indices
        
        return result_mask
    
    def generate_quality_report(self) -> Dict:
        report = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.assess_missing_values(),
            'duplicates': self.identify_duplicates(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        return report


class RFMAnalyzer:
    
    def __init__(self, transaction_df: pd.DataFrame, client_col: str, date_col: str, value_col: str):
        self.transaction_df = transaction_df
        self.client_col = client_col
        self.date_col = date_col
        self.value_col = value_col
        self.reference_date = None
        
    def calculate_metrics(self, reference_date=None) -> pd.DataFrame:
        if reference_date is None:
            self.reference_date = self.transaction_df[self.date_col].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = reference_date
        
        rfm_data = self.transaction_df.groupby(self.client_col).agg({
            self.date_col: lambda x: (self.reference_date - x.max()).days,
            self.client_col: 'count',
            self.value_col: 'sum'
        })
        
        rfm_data.columns = ['recency_days', 'frequency_count', 'monetary_total']
        rfm_data = rfm_data.reset_index()
        
        return rfm_data
    
    def assign_scores(self, rfm_df: pd.DataFrame, quartiles: int = 4) -> pd.DataFrame:
        rfm_df['recency_score'] = pd.qcut(
            rfm_df['recency_days'], 
            q=quartiles, 
            labels=range(quartiles, 0, -1),
            duplicates='drop'
        ).astype(int)
        
        rfm_df['frequency_score'] = pd.qcut(
            rfm_df['frequency_count'],
            q=quartiles,
            labels=range(1, quartiles + 1),
            duplicates='drop'
        ).astype(int)
        
        rfm_df['monetary_score'] = pd.qcut(
            rfm_df['monetary_total'],
            q=quartiles,
            labels=range(1, quartiles + 1),
            duplicates='drop'
        ).astype(int)
        
        rfm_df['combined_score'] = (
            rfm_df['recency_score'] +
            rfm_df['frequency_score'] +
            rfm_df['monetary_score']
        )
        
        rfm_df['rfm_segment'] = (
            rfm_df['recency_score'].astype(str) +
            rfm_df['frequency_score'].astype(str) +
            rfm_df['monetary_score'].astype(str)
        )
        
        return rfm_df
    
    def create_segments(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        segment_mapping = []
        
        for _, row in rfm_df.iterrows():
            r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
            
            if r >= 3 and f >= 3 and m >= 3:
                segment = 'Champions'
            elif r >= 3 and f >= 2:
                segment = 'Loyal'
            elif r >= 3:
                segment = 'Potential'
            elif r == 2 and f >= 2:
                segment = 'Needs Attention'
            elif r == 2:
                segment = 'About to Sleep'
            else:
                segment = 'At Risk'
            
            segment_mapping.append(segment)
        
        rfm_df['segment_label'] = segment_mapping
        return rfm_df


class FeatureEngineer:
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        df_copy['transaction_year'] = df_copy[date_column].dt.year
        df_copy['transaction_month'] = df_copy[date_column].dt.month
        df_copy['transaction_day'] = df_copy[date_column].dt.day
        df_copy['transaction_dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy['transaction_hour'] = df_copy[date_column].dt.hour
        df_copy['transaction_quarter'] = df_copy[date_column].dt.quarter
        df_copy['is_weekend'] = df_copy['transaction_dayofweek'].isin([5, 6]).astype(int)
        
        return df_copy
    
    @staticmethod
    def aggregate_by_category(df: pd.DataFrame, 
                             group_col: str, 
                             category_col: str, 
                             value_col: str) -> pd.DataFrame:
        category_pivot = df.pivot_table(
            index=group_col,
            columns=category_col,
            values=value_col,
            aggfunc='sum',
            fill_value=0
        )
        
        category_pivot.columns = [f'category_{col}_total' for col in category_pivot.columns]
        return category_pivot
    
    @staticmethod
    def calculate_derived_metrics(df: pd.DataFrame, 
                                  quantity_col: str, 
                                  price_col: str) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['total_value'] = df_copy[quantity_col] * df_copy[price_col]
        df_copy['average_item_price'] = df_copy['total_value'] / df_copy[quantity_col]
        
        return df_copy
    
    @staticmethod
    def apply_log_transformation(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_transformed = df.copy()
        
        for col in columns:
            df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
        
        return df_transformed


class ClusterAnalyzer:
    
    def __init__(self, data: np.ndarray):
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = None
        
    def preprocess_data(self, apply_scaling: bool = True) -> np.ndarray:
        if apply_scaling:
            self.scaled_data = self.scaler.fit_transform(self.data)
            return self.scaled_data
        return self.data
    
    def calculate_inertia(self, cluster_range: range) -> List[float]:
        from sklearn.cluster import KMeans
        
        inertia_values = []
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data if self.scaled_data is not None else self.data)
            inertia_values.append(kmeans.inertia_)
        
        return inertia_values
    
    def calculate_silhouette_scores(self, cluster_range: range) -> List[float]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        silhouette_values = []
        data_to_use = self.scaled_data if self.scaled_data is not None else self.data
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_to_use)
            silhouette_avg = silhouette_score(data_to_use, cluster_labels)
            silhouette_values.append(silhouette_avg)
        
        return silhouette_values
    
    def determine_optimal_clusters(self, cluster_range: range) -> Tuple[int, float]:
        silhouette_scores = self.calculate_silhouette_scores(cluster_range)
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        optimal_score = max(silhouette_scores)
        
        return optimal_k, optimal_score
    
    def fit_kmeans(self, n_clusters: int):
        from sklearn.cluster import KMeans
        
        data_to_use = self.scaled_data if self.scaled_data is not None else self.data
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans_model.fit_predict(data_to_use)
        
        return cluster_assignments, kmeans_model


class TextProcessor:
    
    @staticmethod
    def clean_text(text: str) -> str:
        import re
        
        if pd.isna(text):
            return ''
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def remove_stopwords(text: str, language: str = 'english') -> str:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        if not text:
            return ''
        
        stop_words = set(stopwords.words(language))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_tokens)
    
    @staticmethod
    def process_text_pipeline(text: str) -> str:
        cleaned = TextProcessor.clean_text(text)
        processed = TextProcessor.remove_stopwords(cleaned)
        return processed


class VisualizationHelper:
    
    @staticmethod
    def create_distribution_summary(df: pd.DataFrame, column: str) -> Dict:
        summary = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'q25': df[column].quantile(0.25),
            'q75': df[column].quantile(0.75),
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurtosis()
        }
        return summary
    
    @staticmethod
    def prepare_cluster_summary(df: pd.DataFrame, 
                               cluster_col: str, 
                               numeric_cols: List[str]) -> pd.DataFrame:
        summary_df = df.groupby(cluster_col)[numeric_cols].agg(['mean', 'median', 'std', 'count'])
        return summary_df.round(2)


def export_results(dataframe: pd.DataFrame, 
                  filename: str, 
                  include_timestamp: bool = True) -> str:
    from datetime import datetime
    
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename}_{timestamp}.csv"
    else:
        filename = f"{filename}.csv"
    
    dataframe.to_csv(filename, index=False)
    return filename


def load_and_validate_data(filepath: str, required_columns: List[str]) -> pd.DataFrame:
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
