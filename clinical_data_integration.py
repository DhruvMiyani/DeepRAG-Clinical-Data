"""
Clinical Data Integration for DeepRAG Pipeline
Connects real clinical CSV data to the vector store
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from utils import ClinicalDataProcessor, FileManager

logger = logging.getLogger(__name__)


class ClinicalDataIntegrator:
    """Integrates real clinical CSV data into DeepRAG pipeline"""
    
    def __init__(self, data_path: str = "./"):
        """Initialize with path to clinical data files"""
        self.data_path = data_path
        self.datasets = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_clinical_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available clinical datasets"""
        dataset_files = {
            # HAPI datasets
            'hapi_train_chronologies': 'train.chronologies.csv',
            'hapi_devel_chronologies': 'devel.chronologies.csv', 
            'hapi_test_chronologies': 'test.chronologies.csv',
            'hapi_train_labels': 'train.labels.csv',
            'hapi_devel_labels': 'devel.labels.csv',
            'hapi_test_labels': 'test.labels.csv',
            'hapi_devel_admittimes': 'devel.admittimes.csv',
            'hapi_test_admittimes': 'test.admittimes.csv',
            'hapi_negative_labels': 'negative_labels.csv'
        }
        
        loaded_datasets = {}
        
        for dataset_name, filename in dataset_files.items():
            try:
                filepath = f"{self.data_path}/{filename}"
                df = pd.read_csv(filepath)
                loaded_datasets[dataset_name] = df
                self.logger.info(f"Loaded {dataset_name}: {len(df)} records")
            except FileNotFoundError:
                self.logger.warning(f"File not found: {filepath}")
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")
        
        self.datasets = loaded_datasets
        return loaded_datasets
    
    def create_clinical_documents(self) -> List[Document]:
        """Convert clinical data to documents for vector store"""
        documents = []
        
        # Process chronologies data
        for dataset_name in ['hapi_train_chronologies', 'hapi_devel_chronologies']:
            if dataset_name in self.datasets:
                docs = self._process_chronologies(self.datasets[dataset_name], dataset_name)
                documents.extend(docs)
        
        # Process admissions data
        for dataset_name in ['hapi_devel_admittimes', 'hapi_test_admittimes']:
            if dataset_name in self.datasets:
                docs = self._process_admissions(self.datasets[dataset_name], dataset_name)
                documents.extend(docs)
        
        # Process labels data
        for dataset_name in ['hapi_train_labels', 'hapi_devel_labels']:
            if dataset_name in self.datasets:
                docs = self._process_labels(self.datasets[dataset_name], dataset_name)
                documents.extend(docs)
        
        self.logger.info(f"Created {len(documents)} clinical documents")
        return documents
    
    def _process_chronologies(self, df: pd.DataFrame, source: str) -> List[Document]:
        """Process chronologies data into documents"""
        documents = []
        
        for _, row in df.iterrows():
            # Create document from each patient chronology
            content = self._format_chronology_record(row)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': source,
                    'subject_id': row.get('subject_id', 'unknown'),
                    'hadm_id': row.get('hadm_id', 'unknown'),
                    'timestamp': row.get('timestamp', 'unknown'),
                    'record_type': 'chronology'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_admissions(self, df: pd.DataFrame, source: str) -> List[Document]:
        """Process admission times data into documents"""
        documents = []
        
        for _, row in df.iterrows():
            content = self._format_admission_record(row)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': source,
                    'subject_id': row.get('subject_id', 'unknown'),
                    'hadm_id': row.get('hadm_id', 'unknown'),
                    'admittime': row.get('admittime', 'unknown'),
                    'record_type': 'admission'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_labels(self, df: pd.DataFrame, source: str) -> List[Document]:
        """Process labels data into documents"""
        documents = []
        
        for _, row in df.iterrows():
            content = self._format_label_record(row)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': source,
                    'subject_id': row.get('subject_id', 'unknown'),
                    'hadm_id': row.get('hadm_id', 'unknown'),
                    'record_type': 'label'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _format_chronology_record(self, row) -> str:
        """Format chronology record into readable text"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')
        timestamp = row.get('timestamp', 'Unknown')
        observations = row.get('observations', 'No observations')
        
        return f"""Patient Record - Chronology
Subject ID: {subject_id}
Hospital Admission ID: {hadm_id} 
Timestamp: {timestamp}
Clinical Observations: {observations}

This record shows clinical observations for patient {subject_id} during hospital admission {hadm_id} at timestamp {timestamp}. The observations include medical codes that indicate specific clinical conditions and assessments."""
    
    def _format_admission_record(self, row) -> str:
        """Format admission record into readable text"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')
        admittime = row.get('admittime', 'Unknown')
        
        return f"""Hospital Admission Record
Subject ID: {subject_id}
Hospital Admission ID: {hadm_id}
Admission Time: {admittime}

This admission record tracks when patient {subject_id} was admitted to the hospital under admission ID {hadm_id}. The admission time {admittime} is critical for determining if conditions are hospital-acquired (occurring after 48 hours of admission)."""
    
    def _format_label_record(self, row) -> str:
        """Format label record into readable text"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')
        
        # Handle different possible label columns
        label_info = []
        for col in row.index:
            if col not in ['subject_id', 'hadm_id'] and pd.notna(row[col]):
                label_info.append(f"{col}: {row[col]}")
        
        label_text = "; ".join(label_info) if label_info else "No labels"
        
        return f"""Clinical Labels Record
Subject ID: {subject_id}
Hospital Admission ID: {hadm_id}
Labels: {label_text}

This record contains clinical outcome labels for patient {subject_id} during admission {hadm_id}. These labels indicate diagnosed conditions, particularly hospital-acquired conditions like pressure injuries."""
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded datasets"""
        summary = {
            'total_datasets': len(self.datasets),
            'datasets': {}
        }
        
        for name, df in self.datasets.items():
            summary['datasets'][name] = {
                'records': len(df),
                'columns': list(df.columns),
                'sample_subject_ids': df['subject_id'].unique()[:5].tolist() if 'subject_id' in df.columns else []
            }
        
        return summary
    
    def create_enhanced_knowledge_base(self) -> List[Document]:
        """Create enhanced knowledge base combining real data with clinical knowledge"""
        documents = []
        
        # Add clinical domain knowledge
        clinical_knowledge = [
            """Hospital-acquired pressure injuries (HAPI) are localized damage to the skin and underlying soft tissue, usually over a bony prominence. The injury occurs as a result of pressure or pressure in combination with shear. Common risk factors include immobility, poor nutrition, and moisture. Prevention strategies include regular repositioning, proper nutrition, and use of pressure-relieving surfaces.""",
            
            """Clinical observation code C0392747 refers to pressure ulcer assessment. This includes evaluation of wound size, depth, tissue type, exudate, and surrounding skin condition. The Braden Scale is commonly used to assess pressure injury risk, with scores ranging from 6-23, where lower scores indicate higher risk.""",
            
            """Patient admission timestamps are critical for tracking hospital-acquired conditions. HADM_ID (Hospital Admission ID) uniquely identifies each hospital admission. Observations recorded after 48 hours of admission are considered hospital-acquired. Common HAI codes include C0392747 (pressure injury), C0684224 (surgical site infection), and C3273238 (catheter-associated UTI)."""
        ]
        
        for i, knowledge in enumerate(clinical_knowledge):
            documents.append(Document(
                page_content=knowledge,
                metadata={'source': 'clinical_knowledge', 'doc_id': i, 'record_type': 'knowledge'}
            ))
        
        # Add real clinical data
        if self.datasets:
            clinical_docs = self.create_clinical_documents()
            documents.extend(clinical_docs)
        
        return documents


def integrate_clinical_data_to_pipeline(pipeline, data_path: str = "./"):
    """Integrate real clinical data into existing DeepRAG pipeline"""
    logger.info("Integrating real clinical data into DeepRAG pipeline")
    
    # Initialize integrator
    integrator = ClinicalDataIntegrator(data_path)
    
    # Load datasets
    datasets = integrator.load_clinical_datasets()
    
    if not datasets:
        logger.warning("No clinical datasets found, using synthetic data")
        return pipeline
    
    # Create enhanced documents
    documents = integrator.create_enhanced_knowledge_base()
    
    # Update pipeline's vector store
    if documents:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks from real clinical data")
        
        # Create new vector store with real data
        pipeline.vector_store = FAISS.from_documents(chunks, pipeline.embeddings)
        pipeline.retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Update DeepRAG core
        pipeline.deeprag_core.retriever = pipeline.retriever
        
        logger.info("Successfully integrated real clinical data into pipeline")
        
        # Log summary
        summary = integrator.get_dataset_summary()
        logger.info(f"Dataset summary: {summary}")
    
    return pipeline