import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class FileManager:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists, create if not."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {str(e)}")
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """Save object to pickle file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Object saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving pickle to {filepath}: {str(e)}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Optional[Any]:
        """Load object from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {str(e)}")
            return None


class DataProcessor:
    """Utility class for data processing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters if needed
        # Add more cleaning logic as required
        
        return text.strip()
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_length:
                break
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def merge_dataframes(
        dfs: List[pd.DataFrame],
        how: str = 'outer',
        on: Optional[str] = None
    ) -> pd.DataFrame:
        """Merge multiple dataframes."""
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return dfs[0]
        
        result = dfs[0]
        for df in dfs[1:]:
            if on:
                result = pd.merge(result, df, how=how, on=on)
            else:
                result = pd.concat([result, df], ignore_index=True)
        
        return result


class MetricsCalculator:
    """Utility class for calculating various metrics."""
    
    @staticmethod
    def precision(true_positives: int, false_positives: int) -> float:
        """Calculate precision."""
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)
    
    @staticmethod
    def recall(true_positives: int, false_negatives: int) -> float:
        """Calculate recall."""
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def accuracy(
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int
    ) -> float:
        """Calculate accuracy."""
        total = true_positives + true_negatives + false_positives + false_negatives
        if total == 0:
            return 0.0
        return (true_positives + true_negatives) / total


class ClinicalDataProcessor:
    """Specialized processor for clinical data."""
    
    @staticmethod
    def parse_observations(observations: str) -> List[str]:
        """Parse observation codes from string."""
        if not observations:
            return []
        
        # Assuming observations are space-separated codes
        return observations.strip().split()
    
    @staticmethod
    def format_admission_record(
        subject_id: str,
        hadm_id: str,
        timestamp: str,
        observations: str
    ) -> Dict[str, Any]:
        """Format admission record for processing."""
        return {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'timestamp': timestamp,
            'observations': ClinicalDataProcessor.parse_observations(observations),
            'observation_count': len(ClinicalDataProcessor.parse_observations(observations))
        }
    
    @staticmethod
    def aggregate_patient_data(
        df: pd.DataFrame,
        patient_id_col: str = 'subject_id'
    ) -> pd.DataFrame:
        """Aggregate data by patient."""
        try:
            aggregated = df.groupby(patient_id_col).agg({
                'hadm_id': 'count',
                'observations': lambda x: ' '.join(x) if isinstance(x.iloc[0], str) else ' '.join([' '.join(obs) for obs in x])
            }).reset_index()
            
            aggregated.columns = [patient_id_col, 'admission_count', 'all_observations']
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating patient data: {str(e)}")
            return pd.DataFrame()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )


if __name__ == "__main__":
    # Example usage
    setup_logging()
    logger.info("Utilities module loaded successfully")