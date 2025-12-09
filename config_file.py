from pathlib import Path
from typing import Dict, List

class ProjectConfiguration:
    
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output_graphs'
    ASSETS_DIR = BASE_DIR / 'visualization_assets'
    
    DATA_FILE = 'transaction_records.xlsx'
    SHEET_NAME = 'Online Retail'
    
    COLUMN_MAPPING = {
        'InvoiceNo': 'TransactionReference',
        'StockCode': 'ProductSKU',
        'Description': 'ItemDescription',
        'Quantity': 'OrderQuantity',
        'InvoiceDate': 'TransactionDate',
        'UnitPrice': 'PricePerUnit',
        'CustomerID': 'ClientIdentifier',
        'Country': 'ClientLocation'
    }
    
    NUMERIC_COLUMNS = ['OrderQuantity', 'PricePerUnit']
    CATEGORICAL_COLUMNS = ['ProductSKU', 'ItemDescription', 'ClientLocation']
    DATE_COLUMNS = ['TransactionDate']
    
    RFM_CONFIG = {
        'recency_column': 'DaysSinceLastOrder',
        'frequency_column': 'TransactionCount',
        'monetary_column': 'TotalSpending',
        'quartile_count': 4
    }
    
    CLUSTERING_CONFIG = {
        'min_clusters': 2,
        'max_clusters': 10,
        'random_state': 42,
        'n_init': 10,
        'optimal_clusters': 3
    }
    
    PCA_CONFIG = {
        'variance_threshold': 0.95,
        'random_state': 42
    }
    
    PRODUCT_CLUSTERING_CONFIG = {
        'max_features': 100,
        'optimal_clusters': 5,
        'random_state': 42
    }
    
    SEGMENT_LABELS = {
        0: 'High Value Clients',
        1: 'Medium Value Clients', 
        2: 'Low Value Clients'
    }
    
    VISUALIZATION_CONFIG = {
        'figure_size': (12, 6),
        'dpi': 100,
        'style': 'whitegrid',
        'color_palette': 'viridis',
        'font_size': 12
    }
    
    OUTPUT_FILES = {
        'segmentation_results': 'client_segmentation_results.csv',
        'cluster_profiles': 'cluster_behavioral_profiles.csv',
        'rfm_scores': 'rfm_analysis_scores.csv'
    }
    
    @classmethod
    def get_data_path(cls) -> Path:
        return cls.DATA_DIR / cls.DATA_FILE
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        return cls.OUTPUT_DIR / filename
    
    @classmethod
    def ensure_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.ASSETS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_column_name(cls, original_name: str) -> str:
        return cls.COLUMN_MAPPING.get(original_name, original_name)


class AnalysisParameters:
    
    OUTLIER_METHODS = {
        'iqr': {
            'multiplier': 1.5,
            'description': 'Interquartile Range method'
        },
        'zscore': {
            'threshold': 3.0,
            'description': 'Z-score statistical method'
        }
    }
    
    TEXT_PROCESSING = {
        'min_word_length': 2,
        'language': 'english',
        'remove_numbers': True,
        'lowercase': True
    }
    
    FEATURE_ENGINEERING = {
        'create_temporal': True,
        'create_aggregates': True,
        'apply_log_transform': True,
        'scaling_method': 'standard'
    }
    
    VALIDATION_RULES = {
        'min_transaction_value': 0.01,
        'max_transaction_value': 100000,
        'min_quantity': 1,
        'max_quantity': 10000
    }


class BusinessMetrics:
    
    SEGMENT_DEFINITIONS = {
        'Champions': {
            'recency_min': 3,
            'frequency_min': 3,
            'monetary_min': 3,
            'description': 'Best customers with recent, frequent, and high-value purchases'
        },
        'Loyal': {
            'recency_min': 3,
            'frequency_min': 2,
            'monetary_min': 2,
            'description': 'Consistent customers with regular purchase patterns'
        },
        'Potential': {
            'recency_min': 3,
            'frequency_min': 1,
            'monetary_min': 1,
            'description': 'Recent customers with growth potential'
        },
        'At Risk': {
            'recency_max': 2,
            'frequency_max': 2,
            'monetary_max': 2,
            'description': 'Customers showing signs of churn'
        }
    }
    
    KPI_THRESHOLDS = {
        'high_value_spending': 1000,
        'frequent_buyer_transactions': 10,
        'recent_activity_days': 30,
        'churn_risk_days': 180
    }
    
    @classmethod
    def calculate_customer_lifetime_value(cls, 
                                         avg_transaction: float,
                                         frequency: int,
                                         retention_rate: float = 0.75) -> float:
        return (avg_transaction * frequency) / (1 - retention_rate)


class DataValidation:
    
    REQUIRED_COLUMNS = [
        'TransactionReference',
        'ProductSKU', 
        'ItemDescription',
        'OrderQuantity',
        'TransactionDate',
        'PricePerUnit',
        'ClientIdentifier',
        'ClientLocation'
    ]
    
    DATA_TYPES = {
        'TransactionReference': 'object',
        'ProductSKU': 'object',
        'ItemDescription': 'object',
        'OrderQuantity': 'float64',
        'TransactionDate': 'datetime64[ns]',
        'PricePerUnit': 'float64',
        'ClientIdentifier': 'float64',
        'ClientLocation': 'object'
    }
    
    @staticmethod
    def validate_dataframe(df, required_cols: List[str]) -> Dict[str, bool]:
        validation_results = {
            'has_required_columns': all(col in df.columns for col in required_cols),
            'has_duplicates': df.duplicated().any(),
            'has_nulls': df.isnull().any().any(),
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        return validation_results


if __name__ == '__main__':
    ProjectConfiguration.ensure_directories()
    print("Configuration loaded successfully")
    print(f"Base directory: {ProjectConfiguration.BASE_DIR}")
    print(f"Data path: {ProjectConfiguration.get_data_path()}")
