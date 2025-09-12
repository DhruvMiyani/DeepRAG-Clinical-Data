"""
Integration module to connect real MIMIC-III nosocomial data to DeepRAG pipeline
"""

import logging
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import Document

from datasets import NosocomialDataLoader
from deeprag_pipeline import DeepRAGPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class MimicDeepRAGIntegrator:
    """Integrates real MIMIC-III nosocomial data with DeepRAG pipeline"""
    
    def __init__(self, data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0"):
        """Initialize integrator with path to MIMIC data"""
        self.data_path = data_path
        self.loader = NosocomialDataLoader(data_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_and_convert_data(self, conditions: List[str] = ['hapi']) -> List[Document]:
        """Load MIMIC data and convert to documents for RAG"""
        self.logger.info(f"Loading MIMIC-III data for conditions: {conditions}")
        
        documents = []
        
        for condition in conditions:
            self.logger.info(f"Processing {condition.upper()} datasets...")
            
            # Load datasets for this condition
            datasets = self.loader.load_condition_datasets(condition)
            
            if not datasets:
                self.logger.warning(f"No datasets found for condition: {condition}")
                continue
            
            # Convert each dataset type to documents
            docs = self._convert_datasets_to_documents(datasets, condition)
            documents.extend(docs)
            
            self.logger.info(f"Created {len(docs)} documents from {condition} data")
        
        self.logger.info(f"Total documents created: {len(documents)}")
        return documents
    
    def _convert_datasets_to_documents(
        self, 
        datasets: Dict[str, pd.DataFrame], 
        condition: str
    ) -> List[Document]:
        """Convert datasets to LangChain documents"""
        documents = []
        
        # Process chronologies (patient timelines)
        for split in ['train', 'devel', 'test']:
            chronology_key = f"{split}_chronologies"
            if chronology_key in datasets:
                docs = self._process_chronologies(
                    datasets[chronology_key], 
                    condition, 
                    split
                )
                documents.extend(docs)
        
        # Process admission times
        for split in ['devel', 'test']:
            admit_key = f"{split}_admittimes"
            if admit_key in datasets:
                docs = self._process_admissions(
                    datasets[admit_key], 
                    condition, 
                    split
                )
                documents.extend(docs)
        
        # Process labels
        for split in ['train', 'devel', 'test']:
            label_key = f"{split}_labels"
            if label_key in datasets:
                docs = self._process_labels(
                    datasets[label_key], 
                    condition, 
                    split
                )
                documents.extend(docs)
        
        return documents
    
    def _process_chronologies(
        self, 
        df: pd.DataFrame, 
        condition: str, 
        split: str
    ) -> List[Document]:
        """Convert chronologies to documents"""
        documents = []
        
        for _, row in df.iterrows():
            # Create comprehensive patient chronology document
            content = self._format_patient_chronology(row, condition)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': f"{condition}_{split}_chronologies",
                    'condition': condition,
                    'split': split,
                    'subject_id': str(row.get('subject_id', 'unknown')),
                    'hadm_id': str(row.get('hadm_id', 'unknown')), 
                    'timestamp': str(row.get('timestamp', 'unknown')),
                    'record_type': 'chronology'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_admissions(
        self, 
        df: pd.DataFrame, 
        condition: str, 
        split: str
    ) -> List[Document]:
        """Convert admission data to documents"""
        documents = []
        
        for _, row in df.iterrows():
            content = self._format_admission_record(row, condition)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': f"{condition}_{split}_admissions",
                    'condition': condition,
                    'split': split, 
                    'subject_id': str(row.get('subject_id', 'unknown')),
                    'hadm_id': str(row.get('hadm_id', 'unknown')),
                    'admittime': str(row.get('admittime', 'unknown')),
                    'record_type': 'admission'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_labels(
        self, 
        df: pd.DataFrame, 
        condition: str, 
        split: str
    ) -> List[Document]:
        """Convert labels to documents"""
        documents = []
        
        for _, row in df.iterrows():
            content = self._format_label_record(row, condition)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': f"{condition}_{split}_labels",
                    'condition': condition,
                    'split': split,
                    'subject_id': str(row.get('subject_id', 'unknown')),
                    'hadm_id': str(row.get('hadm_id', 'unknown')),
                    'record_type': 'label'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _format_patient_chronology(self, row, condition: str) -> str:
        """Format patient chronology into readable clinical text"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')  
        timestamp = row.get('timestamp', 'Unknown')
        observations = row.get('observations', 'No observations recorded')
        
        # Map condition to clinical context
        condition_context = {
            'hapi': 'Hospital-Acquired Pressure Injury',
            'haaki': 'Hospital-Acquired Acute Kidney Injury', 
            'haa': 'Hospital-Acquired Anemia'
        }
        
        condition_name = condition_context.get(condition, condition.upper())
        
        return f"""Clinical Patient Chronology - {condition_name}

Patient Information:
- Subject ID: {subject_id}
- Hospital Admission ID: {hadm_id}
- Timestamp: {timestamp}

Clinical Observations and Codes: {observations}

This chronological record documents clinical observations for patient {subject_id} during hospital admission {hadm_id}. The timestamp {timestamp} indicates when these observations were recorded. The clinical observation codes represent specific medical conditions, assessments, and interventions relevant to {condition_name} risk assessment and management.

For {condition_name} analysis, these temporal observations help identify patterns and risk factors that may lead to hospital-acquired conditions developing after 48 hours of admission."""

    def _format_admission_record(self, row, condition: str) -> str:
        """Format admission record for clinical context"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')
        admittime = row.get('admittime', 'Unknown')
        
        return f"""Hospital Admission Record

Patient Demographics:
- Subject ID: {subject_id} 
- Hospital Admission ID: {hadm_id}
- Admission Date/Time: {admittime}

Clinical Significance:
This admission record establishes the baseline for hospital-acquired condition tracking. Patient {subject_id} was admitted to the hospital on {admittime} under admission ID {hadm_id}. 

Hospital-acquired conditions are defined as clinical conditions that develop after 48 hours of hospital admission. The admission timestamp {admittime} serves as the reference point for determining if subsequent clinical observations and diagnoses qualify as hospital-acquired rather than present-on-admission conditions.

This temporal baseline is critical for {condition} risk assessment and prevention protocols."""

    def _format_label_record(self, row, condition: str) -> str:
        """Format label record with clinical outcomes"""
        subject_id = row.get('subject_id', 'Unknown')
        hadm_id = row.get('hadm_id', 'Unknown')
        
        # Extract label information (assuming last column contains the outcome)
        label_cols = [col for col in row.index if col not in ['subject_id', 'hadm_id']]
        outcome_info = []
        
        for col in label_cols:
            if pd.notna(row[col]):
                outcome_info.append(f"{col}: {row[col]}")
        
        outcome_text = "; ".join(outcome_info) if outcome_info else "No outcomes recorded"
        
        return f"""Clinical Outcome Labels

Patient Case:
- Subject ID: {subject_id}
- Hospital Admission ID: {hadm_id}
- Condition Category: {condition.upper()}

Outcome Labels: {outcome_text}

This record contains validated clinical outcome labels for patient {subject_id} during hospital admission {hadm_id}. These labels represent confirmed diagnoses and clinical outcomes, particularly focusing on hospital-acquired conditions that developed during the hospital stay.

The outcome labels serve as ground truth for predictive modeling and quality improvement initiatives targeting the prevention of hospital-acquired conditions."""

    def integrate_with_deeprag(
        self, 
        pipeline: DeepRAGPipeline,
        conditions: List[str] = ['hapi'],
        chunk_size: int = 750,
        chunk_overlap: int = 100
    ) -> DeepRAGPipeline:
        """Integrate real MIMIC data with existing DeepRAG pipeline"""
        self.logger.info("Integrating real MIMIC-III data with DeepRAG pipeline")
        
        # Load and convert MIMIC data
        documents = self.load_and_convert_data(conditions)
        
        if not documents:
            self.logger.error("No documents created from MIMIC data")
            return pipeline
        
        # Add clinical knowledge documents
        clinical_knowledge = self._get_clinical_knowledge_docs()
        documents.extend(clinical_knowledge)
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} text chunks from {len(documents)} documents")
        
        # Create new vector store with real data
        pipeline.vector_store = FAISS.from_documents(chunks, pipeline.embeddings)
        pipeline.retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 6})
        
        # Update DeepRAG core
        pipeline.deeprag_core.retriever = pipeline.retriever
        
        self.logger.info("Successfully integrated real MIMIC-III data into DeepRAG pipeline")
        
        return pipeline
    
    def _get_clinical_knowledge_docs(self) -> List[Document]:
        """Add expert clinical knowledge to complement real data"""
        knowledge_texts = [
            """Hospital-acquired pressure injuries (HAPI) are localized damage to the skin and underlying soft tissue, usually over a bony prominence. The injury occurs as a result of pressure or pressure in combination with shear. Common risk factors include immobility, poor nutrition, and moisture. Prevention strategies include regular repositioning, proper nutrition, and use of pressure-relieving surfaces.""",
            
            """Clinical observation code C0392747 refers to pressure ulcer assessment. This includes evaluation of wound size, depth, tissue type, exudate, and surrounding skin condition. The Braden Scale is commonly used to assess pressure injury risk, with scores ranging from 6-23, where lower scores indicate higher risk.""",
            
            """Patient admission timestamps are critical for tracking hospital-acquired conditions. HADM_ID (Hospital Admission ID) uniquely identifies each hospital admission. Observations recorded after 48 hours of admission are considered hospital-acquired. Common HAI codes include C0392747 (pressure injury), C0684224 (surgical site infection), and C3273238 (catheter-associated UTI)."""
        ]
        
        docs = []
        for i, text in enumerate(knowledge_texts):
            docs.append(Document(
                page_content=text,
                metadata={
                    'source': 'clinical_knowledge',
                    'doc_id': i,
                    'record_type': 'knowledge'
                }
            ))
        
        return docs


def integrate_mimic_data_to_deeprag(
    data_path: str = "./nosocomial-risk-datasets-from-mimic-iii-1.0",
    conditions: List[str] = ['hapi']
) -> DeepRAGPipeline:
    """
    Convenience function to create DeepRAG pipeline with real MIMIC data
    
    Args:
        data_path: Path to nosocomial dataset directory
        conditions: List of conditions to include ('hapi', 'haaki', 'haa')
    
    Returns:
        DeepRAG pipeline with real MIMIC data integrated
    """
    # Initialize pipeline
    pipeline = DeepRAGPipeline()
    
    # Initialize integrator
    integrator = MimicDeepRAGIntegrator(data_path)
    
    # Integrate real data
    pipeline = integrator.integrate_with_deeprag(pipeline, conditions)
    
    return pipeline


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline with real MIMIC data
    pipeline = integrate_mimic_data_to_deeprag(
        conditions=['hapi']  # Start with pressure injury data
    )
    
    # Test with a clinical question
    result = pipeline.process_question(
        "What are the risk factors for hospital-acquired pressure injuries in patient 12?",
        use_deeprag=True
    )
    
    print("Integration test completed!")
    print(f"Answer: {result.get('answer', 'No answer generated')}")