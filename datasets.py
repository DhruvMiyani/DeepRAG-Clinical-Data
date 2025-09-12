"""
Nosocomial Risk Dataset Loader for MIMIC-III Derived Data
Loads hospital-acquired condition datasets: HAPI, HAAKI, and HAA
"""

import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class NosocomialDataLoader:
    """Loader for MIMIC-III derived nosocomial risk datasets"""
    
    def __init__(self, data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0"):
        """
        Initialize dataset loader
        
        Args:
            data_path: Path to directory containing CSV files
        """
        self.data_path = Path(data_path)
        self.datasets = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dataset file mappings
        self.file_mappings = {
            # HAPI - Hospital Acquired Pressure Injury
            'hapi': {
                'train_chronologies': 'train.chronologies.csv',
                'devel_chronologies': 'devel.chronologies.csv', 
                'test_chronologies': 'test.chronologies.csv',
                'train_labels': 'train.labels.csv',
                'devel_labels': 'devel.labels.csv',
                'test_labels': 'test.labels.csv',
                'devel_admittimes': 'devel.admittimes.csv',
                'test_admittimes': 'test.admittimes.csv',
                'negative_labels': 'negative_labels.csv'
            },
            # HAAKI - Hospital Acquired Acute Kidney Injury  
            'haaki': {
                'train_chronologies': 'train.chronologies.csv',
                'devel_chronologies': 'devel.chronologies.csv',
                'test_chronologies': 'test.chronologies.csv', 
                'train_labels': 'train.labels.csv',
                'devel_labels': 'devel.labels.csv',
                'test_labels': 'test.labels.csv',
                'devel_admittimes': 'devel.admittimes.csv',
                'test_admittimes': 'test.admittimes.csv',
                'negative_labels': 'negative_labels.csv'
            },
            # HAA - Hospital Acquired Anemia
            'haa': {
                'train_chronologies': 'train.chronologies.csv',
                'devel_chronologies': 'devel.chronologies.csv',
                'test_chronologies': 'test.chronologies.csv',
                'train_labels': 'train.labels.csv', 
                'devel_labels': 'devel.labels.csv',
                'test_labels': 'test.labels.csv',
                'devel_admittimes': 'devel.admittimes.csv',
                'test_admittimes': 'test.admittimes.csv',
                'negative_labels': 'negative_labels.csv'
            }
        }
        
        # Clinical condition mappings
        self.condition_codes = {
            'hapi': 'C0392747',  # Pressure injury
            'haaki': 'C0022116',  # Acute kidney injury  
            'haa': 'C0002871'     # Anemia
        }
    
    def load_condition_datasets(self, condition: str) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets for a specific condition
        
        Args:
            condition: One of 'hapi', 'haaki', 'haa'
            
        Returns:
            Dictionary of loaded DataFrames
        """
        if condition not in self.file_mappings:
            raise ValueError(f"Unknown condition: {condition}. Must be one of {list(self.file_mappings.keys())}")
        
        datasets = {}
        condition_path = self.data_path / condition
        
        if not condition_path.exists():
            self.logger.warning(f"Condition directory not found: {condition_path}")
            return datasets
        
        file_mapping = self.file_mappings[condition]
        
        for dataset_name, filename in file_mapping.items():
            filepath = condition_path / filename
            
            try:
                df = pd.read_csv(filepath)
                datasets[dataset_name] = df
                self.logger.info(f"Loaded {condition}_{dataset_name}: {len(df)} records")
            except FileNotFoundError:
                self.logger.warning(f"File not found: {filepath}")
            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {e}")
        
        return datasets
    
    def load_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all datasets for all conditions"""
        all_datasets = {}
        
        for condition in self.file_mappings.keys():
            datasets = self.load_condition_datasets(condition)
            if datasets:
                all_datasets[condition] = datasets
        
        self.datasets = all_datasets
        return all_datasets
    
    def get_chronologies(self, condition: str, split: str = 'train') -> Optional[pd.DataFrame]:
        """
        Get chronologies data for specific condition and split
        
        Args:
            condition: 'hapi', 'haaki', or 'haa' 
            split: 'train', 'devel', or 'test'
            
        Returns:
            DataFrame with chronologies data
        """
        if condition not in self.datasets:
            self.load_condition_datasets(condition)
        
        dataset_key = f"{split}_chronologies"
        return self.datasets.get(condition, {}).get(dataset_key)
    
    def get_labels(self, condition: str, split: str = 'train') -> Optional[pd.DataFrame]:
        """Get labels data for specific condition and split"""
        if condition not in self.datasets:
            self.load_condition_datasets(condition) 
            
        dataset_key = f"{split}_labels"
        return self.datasets.get(condition, {}).get(dataset_key)
    
    def get_admittimes(self, condition: str, split: str = 'devel') -> Optional[pd.DataFrame]:
        """Get admission times data for specific condition and split"""
        if condition not in self.datasets:
            self.load_condition_datasets(condition)
            
        dataset_key = f"{split}_admittimes" 
        return self.datasets.get(condition, {}).get(dataset_key)
    
    def get_negative_labels(self, condition: str) -> Optional[pd.DataFrame]:
        """Get negative labels for specific condition"""
        if condition not in self.datasets:
            self.load_condition_datasets(condition)
            
        return self.datasets.get(condition, {}).get('negative_labels')
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics for all loaded datasets"""
        summary = {
            'conditions': list(self.datasets.keys()),
            'total_conditions': len(self.datasets),
            'details': {}
        }
        
        for condition, datasets in self.datasets.items():
            condition_summary = {
                'datasets': list(datasets.keys()),
                'total_records': 0,
                'condition_code': self.condition_codes.get(condition, 'Unknown'),
                'dataset_details': {}
            }
            
            for dataset_name, df in datasets.items():
                dataset_info = {
                    'records': len(df),
                    'columns': list(df.columns),
                    'unique_subjects': df['subject_id'].nunique() if 'subject_id' in df.columns else 0,
                    'unique_admissions': df['hadm_id'].nunique() if 'hadm_id' in df.columns else 0
                }
                
                condition_summary['dataset_details'][dataset_name] = dataset_info
                condition_summary['total_records'] += len(df)
            
            summary['details'][condition] = condition_summary
        
        return summary
    
    def extract_patient_chronology(
        self, 
        condition: str, 
        subject_id: int, 
        split: str = 'train'
    ) -> Optional[pd.DataFrame]:
        """Extract chronology for specific patient"""
        chronologies = self.get_chronologies(condition, split)
        
        if chronologies is None:
            return None
            
        patient_data = chronologies[chronologies['subject_id'] == subject_id]
        return patient_data.sort_values('timestamp') if not patient_data.empty else None
    
    def get_condition_statistics(self, condition: str) -> Dict:
        """Get detailed statistics for a specific condition"""
        if condition not in self.datasets:
            self.load_condition_datasets(condition)
        
        stats = {
            'condition': condition,
            'condition_code': self.condition_codes.get(condition),
            'datasets_loaded': len(self.datasets.get(condition, {})),
            'splits': {}
        }
        
        condition_data = self.datasets.get(condition, {})
        
        for split in ['train', 'devel', 'test']:
            chronologies = condition_data.get(f"{split}_chronologies")
            labels = condition_data.get(f"{split}_labels")
            
            if chronologies is not None or labels is not None:
                split_stats = {}
                
                if chronologies is not None:
                    split_stats['chronologies'] = {
                        'total_records': len(chronologies),
                        'unique_patients': chronologies['subject_id'].nunique(),
                        'unique_admissions': chronologies['hadm_id'].nunique(),
                        'date_range': [
                            chronologies['timestamp'].min(),
                            chronologies['timestamp'].max()
                        ] if 'timestamp' in chronologies.columns else None
                    }
                
                if labels is not None:
                    split_stats['labels'] = {
                        'total_records': len(labels),
                        'positive_cases': len(labels[labels.iloc[:, -1] == 1]) if len(labels.columns) > 2 else 0
                    }
                
                stats['splits'][split] = split_stats
        
        return stats


# Convenience functions for backward compatibility
def load_hapi_datasets(data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0") -> Dict[str, pd.DataFrame]:
    """Load HAPI (Hospital Acquired Pressure Injury) datasets"""
    loader = NosocomialDataLoader(data_path)
    return loader.load_condition_datasets('hapi')


def load_haaki_datasets(data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0") -> Dict[str, pd.DataFrame]:
    """Load HAAKI (Hospital Acquired Acute Kidney Injury) datasets"""
    loader = NosocomialDataLoader(data_path)
    return loader.load_condition_datasets('haaki')


def load_haa_datasets(data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0") -> Dict[str, pd.DataFrame]:
    """Load HAA (Hospital Acquired Anemia) datasets"""
    loader = NosocomialDataLoader(data_path)
    return loader.load_condition_datasets('haa')


def load_all_nosocomial_data(data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0") -> NosocomialDataLoader:
    """Load all nosocomial datasets and return loader instance"""
    loader = NosocomialDataLoader(data_path)
    loader.load_all_datasets()
    return loader


# Main execution for testing
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = NosocomialDataLoader("./nosocomial-risk-datasets-from-mimic-iii-1.0")
    
    # Load all datasets
    all_data = loader.load_all_datasets()
    
    # Print summary
    summary = loader.get_dataset_summary()
    print("Dataset Summary:")
    for condition, details in summary['details'].items():
        print(f"\n{condition.upper()} ({details['condition_code']}):")
        print(f"  Total records: {details['total_records']}")
        print(f"  Datasets: {', '.join(details['datasets'])}")
    
    # Get specific condition statistics
    if 'hapi' in all_data:
        hapi_stats = loader.get_condition_statistics('hapi')
        print(f"\nHAPI Statistics: {hapi_stats}")