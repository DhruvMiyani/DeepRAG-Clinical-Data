"""
DeepRAG Pipeline: Main execution pipeline with clinical data integration
"""

import logging
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config
from deeprag_core import DeepRAGCore, MDPState, MDPAction, DecisionType
from deeprag_training import DeepRAGTrainer
from utils import FileManager, ClinicalDataProcessor, setup_logging
from datasets import NosocomialDataLoader

# Enhanced logging configuration
setup_logging(log_level="DEBUG", log_file="deeprag_full_pipeline.log")
logger = logging.getLogger(__name__)


class DeepRAGPipeline:
    """Complete DeepRAG pipeline for clinical data"""
    
    def __init__(self):
        """Initialize DeepRAG pipeline with all components"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("=" * 80)
        self.logger.info("Initializing DeepRAG Pipeline for Clinical Data")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
        
        # Initialize configuration
        self.config = Config()
        if not self.config.validate_config():
            raise ValueError("Configuration validation failed")
        
        # Initialize LLM
        self.logger.info(f"Initializing LLM: {self.config.DEFAULT_MODEL}")
        self.llm = ChatOpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model=self.config.DEFAULT_MODEL,
            temperature=self.config.DEFAULT_TEMPERATURE
        )
        
        # Initialize embeddings
        self.logger.info("Initializing OpenAI embeddings")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        # Initialize vector store (will be populated later)
        self.vector_store = None
        self.retriever = None
        
        # Initialize DeepRAG components
        self.deeprag_core = None
        self.trainer = None
        
        # Load clinical data automatically
        self.clinical_loader = None
        
        # Metrics tracking
        self.pipeline_metrics = {
            'total_questions_processed': 0,
            'successful_answers': 0,
            'failed_answers': 0,
            'total_retrievals': 0,
            'total_subqueries': 0,
            'average_latency_ms': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Initialize clinical knowledge base with real data
        self.logger.info("Loading MIMIC-III clinical data...")
        self._initialize_clinical_data()
        
        self.logger.info("Pipeline initialization completed")
    
    def _initialize_clinical_data(self):
        """Initialize clinical knowledge base with real MIMIC-III data"""
        try:
            # Initialize data loader
            data_path = "/Users/dhruvmiyani/Downloads/Projects/RAG-On-Clinical-Data/nosocomial-risk-datasets-from-mimic-iii-1.0"
            self.clinical_loader = NosocomialDataLoader(data_path)
            
            # Load all clinical datasets
            all_datasets = self.clinical_loader.load_all_datasets()
            
            # Convert clinical records to documents for vector store
            clinical_documents = self._convert_clinical_data_to_documents(all_datasets)
            
            if clinical_documents:
                self.logger.info(f"Loaded {len(clinical_documents)} clinical documents")
                # Setup vector store with real clinical data
                self._setup_vector_store(clinical_documents)
            else:
                self.logger.warning("No clinical documents loaded, using sample data")
                self._setup_vector_store(None)
                
        except Exception as e:
            self.logger.error(f"Failed to load MIMIC-III clinical data: {e}")
            self.logger.warning("Falling back to sample clinical data")
            self._setup_vector_store(None)
    
    def _convert_clinical_data_to_documents(self, all_datasets: Dict) -> List[str]:
        """Convert MIMIC-III clinical data to text documents for RAG"""
        documents = []
        
        for condition, datasets in all_datasets.items():
            condition_name = condition.upper()
            
            # Process chronologies (main clinical data)
            for split in ['train', 'devel', 'test']:
                chronologies_key = f"{split}_chronologies"
                if chronologies_key in datasets:
                    chronologies = datasets[chronologies_key]
                    
                    # Group by patient and create clinical narratives
                    for subject_id, patient_data in chronologies.groupby('subject_id'):
                        # Sort by timestamp
                        patient_data = patient_data.sort_values('timestamp')
                        
                        # Create patient narrative
                        narrative_parts = []
                        narrative_parts.append(f"Patient ID: {subject_id}")
                        narrative_parts.append(f"Condition: {condition_name}")
                        
                        # Add temporal sequence of events
                        for _, row in patient_data.iterrows():
                            event_text = f"Time: {row['timestamp']}, "
                            
                            # Add available clinical data
                            if 'code' in row and pd.notna(row['code']):
                                event_text += f"Clinical Code: {row['code']}, "
                            if 'hadm_id' in row and pd.notna(row['hadm_id']):
                                event_text += f"Admission ID: {row['hadm_id']}, "
                            if 'value' in row and pd.notna(row['value']):
                                event_text += f"Value: {row['value']}, "
                            if 'text' in row and pd.notna(row['text']):
                                event_text += f"Description: {row['text']}"
                            
                            narrative_parts.append(event_text.rstrip(', '))
                        
                        # Combine into full narrative
                        full_narrative = "\n".join(narrative_parts)
                        documents.append(full_narrative)
                        
                        # Stop if we have enough documents to avoid token limits
                        if len(documents) >= 200:
                            break
                    
                    if len(documents) >= 200:
                        break
                if len(documents) >= 200:
                    break
        
        self.logger.info(f"Converted {len(documents)} clinical records to documents")
        return documents
    
    def _setup_vector_store(self, documents: Optional[List[str]]):
        """Setup vector store with clinical documents"""
        try:
            # Use provided documents or create sample ones
            if documents is None or len(documents) == 0:
                documents = self._create_sample_clinical_documents()
                self.logger.info("Using sample clinical documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            doc_objects = [Document(page_content=doc) for doc in documents]
            chunks = text_splitter.split_documents(doc_objects)
            
            self.logger.info(f"Created {len(chunks)} document chunks from {len(documents)} documents")
            
            # Create vector store
            self.logger.info("Creating FAISS vector store with embeddings")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.RETRIEVER_K}
            )
            
            # For basic RAG, we don't need DeepRAG core
            # Just set it to None to indicate basic RAG mode
            self.deeprag_core = None
            
            self.logger.info("Vector store setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup vector store: {e}")
            raise
    
    def setup_clinical_knowledge_base(
        self,
        clinical_data: Optional[pd.DataFrame] = None,
        documents: Optional[List[str]] = None
    ):
        """Set up clinical knowledge base for retrieval"""
        self.logger.info("Setting up clinical knowledge base")
        
        # Create sample clinical documents if none provided
        if documents is None:
            documents = self._create_sample_clinical_documents()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        doc_objects = [Document(page_content=doc) for doc in documents]
        chunks = text_splitter.split_documents(doc_objects)
        
        self.logger.info(f"Created {len(chunks)} document chunks")
        
        # Create vector store
        self.logger.info("Creating FAISS vector store")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Initialize DeepRAG core with retriever
        self.deeprag_core = DeepRAGCore(
            llm=self.llm,
            retriever=self.retriever,
            config=self.config
        )
        
        self.trainer = DeepRAGTrainer(self.deeprag_core)
        
        self.logger.info("Clinical knowledge base setup completed")
    
    def _create_sample_clinical_documents(self) -> List[str]:
        """Create sample clinical documents for demonstration"""
        documents = [
            """Hospital-acquired pressure injuries (HAPI) are localized damage to the skin and 
            underlying soft tissue, usually over a bony prominence. The injury occurs as a result 
            of pressure or pressure in combination with shear. Common risk factors include 
            immobility, poor nutrition, and moisture. Prevention strategies include regular 
            repositioning, proper nutrition, and use of pressure-relieving surfaces.""",
            
            """Clinical observation code C0392747 refers to pressure ulcer assessment. This includes 
            evaluation of wound size, depth, tissue type, exudate, and surrounding skin condition. 
            The Braden Scale is commonly used to assess pressure injury risk, with scores ranging 
            from 6-23, where lower scores indicate higher risk.""",
            
            """Patient admission timestamps are critical for tracking hospital-acquired conditions. 
            HADM_ID (Hospital Admission ID) uniquely identifies each hospital admission. 
            Observations recorded after 48 hours of admission are considered hospital-acquired. 
            Common HAI codes include C0392747 (pressure injury), C0684224 (surgical site infection), 
            and C3273238 (catheter-associated UTI).""",
            
            """Treatment protocols for pressure injuries depend on stage classification. 
            Stage 1: Intact skin with non-blanchable erythema. 
            Stage 2: Partial-thickness skin loss with exposed dermis. 
            Stage 3: Full-thickness skin loss. 
            Stage 4: Full-thickness skin and tissue loss. 
            Unstageable: Full-thickness skin and tissue loss with obscured extent.""",
            
            """Risk assessment tools for hospital-acquired conditions include the Braden Scale 
            for pressure injuries, CAUTI bundle for urinary tract infections, and VAP bundle 
            for ventilator-associated pneumonia. Regular reassessment is recommended every 
            24-48 hours or with significant change in patient condition.""",
            
            """Quality metrics for hospital-acquired pressure injuries include incidence rate 
            (new cases per 1000 patient days), prevalence rate (existing cases at point in time), 
            and healing rate. CMS considers Stage 3, Stage 4, and unstageable pressure injuries 
            as never events when acquired during hospitalization.""",
            
            """Clinical documentation should include: admission skin assessment within 24 hours, 
            daily skin assessments, Braden Scale scores, preventive interventions implemented, 
            and any changes in skin condition. Photography with patient consent aids in tracking 
            wound progression.""",
            
            """Prevention bundle components: Risk assessment on admission and regularly thereafter, 
            skin assessment daily, moisture management, nutrition optimization (protein intake 
            1.25-1.5 g/kg/day), repositioning every 2 hours, and appropriate support surfaces 
            for at-risk patients."""
        ]
        
        return documents
    
    def process_question(
        self,
        question: str,
        use_deeprag: bool = True,
        log_details: bool = True
    ) -> Dict[str, Any]:
        """Process a question through the DeepRAG pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing Question: {question}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        result = {
            'question': question,
            'method': 'DeepRAG' if use_deeprag else 'Standard RAG',
            'success': False,
            'answer': None,
            'subqueries': [],
            'retrievals': 0,
            'latency_ms': 0,
            'error': None
        }
        
        try:
            # Check if retriever is available
            if not self.retriever:
                raise Exception("Vector store not initialized. No clinical data available for retrieval.")
            
            if use_deeprag and self.deeprag_core is not None:
                # Use DeepRAG approach
                self.logger.info("Using DeepRAG approach with adaptive retrieval")
                
                # Perform binary tree search
                paths = self.deeprag_core.binary_tree_search(question, max_depth=4)
                
                if not paths:
                    raise Exception("No valid paths found")
                
                # Get optimal path
                optimal_path = self.deeprag_core.find_optimal_path(paths)
                
                if optimal_path:
                    result['answer'] = optimal_path.state.final_answer
                    result['retrievals'] = optimal_path.state.retrieval_count
                    result['subqueries'] = [sq for sq, _ in optimal_path.state.subqueries]
                    result['success'] = True
                    
                    # Log detailed trajectory
                    if log_details:
                        self._log_trajectory(optimal_path)
                
            else:
                # Use standard RAG approach
                self.logger.info("Using basic RAG approach (DeepRAG not available)")
                
                # Simple retrieval and answer
                docs = self.retriever.get_relevant_documents(question)
                self.logger.info(f"Retrieved {len(docs)} documents for question")
                
                if not docs:
                    raise Exception("No relevant documents found for the question")
                
                # Create context from retrieved documents
                context_parts = []
                for i, doc in enumerate(docs[:4]):  # Use up to 4 documents
                    context_parts.append(f"Document {i+1}: {doc.page_content}")
                
                context = "\n\n".join(context_parts)
                
                prompt = f"""You are a clinical AI assistant. Answer the following question based on the provided clinical data context. Be accurate and specific.

Question: {question}

Clinical Context:
{context}

Answer:"""
                
                self.logger.info("Generating answer using LLM")
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                result['answer'] = answer
                result['retrievals'] = len(docs)
                result['success'] = True
                
                self.logger.info(f"Generated answer: {answer[:100]}...")
            
            # Calculate latency
            result['latency_ms'] = int((time.time() - start_time) * 1000)
            
            # Update metrics
            self._update_metrics(result)
            
            self.logger.info(f"Question processed successfully in {result['latency_ms']}ms")
            self.logger.info(f"Retrievals used: {result['retrievals']}")
            self.logger.info(f"Answer: {result['answer'][:200]}...")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error processing question: {e}")
            self.logger.error(traceback.format_exc())
            self.pipeline_metrics['failed_answers'] += 1
        
        return result
    
    def _log_trajectory(self, path_node):
        """Log detailed trajectory information"""
        self.logger.info("\n--- Reasoning Trajectory ---")
        trajectory = path_node.get_path()
        
        for i, (state, action) in enumerate(trajectory):
            self.logger.info(f"\nStep {i+1}:")
            
            if action and action.subquery:
                self.logger.info(f"  Subquery: {action.subquery}")
                self.logger.info(f"  Decision: {action.atomic_decision}")
                
                if state.subqueries and i < len(state.subqueries):
                    _, answer = state.subqueries[i]
                    self.logger.info(f"  Answer: {answer[:100]}...")
            
            if action and action.termination_decision == DecisionType.TERMINATE.value:
                self.logger.info(f"  Final Answer: {state.final_answer[:200]}...")
        
        self.logger.info(f"\nTotal Retrievals: {path_node.state.retrieval_count}")
        self.logger.info("--- End Trajectory ---\n")
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update pipeline metrics"""
        self.pipeline_metrics['total_questions_processed'] += 1
        
        if result['success']:
            self.pipeline_metrics['successful_answers'] += 1
        else:
            self.pipeline_metrics['failed_answers'] += 1
        
        self.pipeline_metrics['total_retrievals'] += result.get('retrievals', 0)
        self.pipeline_metrics['total_subqueries'] += len(result.get('subqueries', []))
        
        # Update average latency
        n = self.pipeline_metrics['total_questions_processed']
        prev_avg = self.pipeline_metrics['average_latency_ms']
        new_latency = result.get('latency_ms', 0)
        self.pipeline_metrics['average_latency_ms'] = (prev_avg * (n-1) + new_latency) / n
    
    def run_comparative_analysis(self, test_questions: List[str]):
        """Run comparative analysis between DeepRAG and standard RAG"""
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPARATIVE ANALYSIS: DeepRAG vs Standard RAG")
        self.logger.info("="*80 + "\n")
        
        results = {
            'deeprag': [],
            'standard': []
        }
        
        for question in test_questions:
            # Test with DeepRAG
            self.logger.info(f"\n--- Testing with DeepRAG ---")
            deeprag_result = self.process_question(question, use_deeprag=True)
            results['deeprag'].append(deeprag_result)
            
            # Test with Standard RAG
            self.logger.info(f"\n--- Testing with Standard RAG ---")
            standard_result = self.process_question(question, use_deeprag=False)
            results['standard'].append(standard_result)
        
        # Compute and log comparison metrics
        self._log_comparison_metrics(results)
        
        return results
    
    def _log_comparison_metrics(self, results: Dict[str, List[Dict]]):
        """Log comparison metrics between approaches"""
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPARISON METRICS")
        self.logger.info("="*80)
        
        for approach in ['deeprag', 'standard']:
            approach_results = results[approach]
            
            total_questions = len(approach_results)
            successful = sum(1 for r in approach_results if r['success'])
            avg_retrievals = sum(r['retrievals'] for r in approach_results) / total_questions
            avg_latency = sum(r['latency_ms'] for r in approach_results) / total_questions
            avg_subqueries = sum(len(r['subqueries']) for r in approach_results) / total_questions
            
            self.logger.info(f"\n{approach.upper()} Metrics:")
            self.logger.info(f"  Success Rate: {successful}/{total_questions} ({100*successful/total_questions:.1f}%)")
            self.logger.info(f"  Avg Retrievals: {avg_retrievals:.2f}")
            self.logger.info(f"  Avg Subqueries: {avg_subqueries:.2f}")
            self.logger.info(f"  Avg Latency: {avg_latency:.0f}ms")
        
        # Calculate improvements
        deeprag_retrievals = sum(r['retrievals'] for r in results['deeprag']) / len(results['deeprag'])
        standard_retrievals = sum(r['retrievals'] for r in results['standard']) / len(results['standard'])
        
        retrieval_reduction = (1 - deeprag_retrievals/standard_retrievals) * 100 if standard_retrievals > 0 else 0
        
        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"DeepRAG Retrieval Reduction: {retrieval_reduction:.1f}%")
        self.logger.info(f"{'='*40}")
    
    def save_results(self, results: Dict[str, Any], filepath: str = "deeprag_results.json"):
        """Save results to file"""
        # Add pipeline metrics to results
        results['pipeline_metrics'] = self.pipeline_metrics
        results['timestamp'] = datetime.now().isoformat()
        results['config'] = {
            'model': self.config.DEFAULT_MODEL,
            'temperature': self.config.DEFAULT_TEMPERATURE,
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP
        }
        
        FileManager.save_json(results, filepath)
        self.logger.info(f"Results saved to {filepath}")
    
    def print_final_summary(self):
        """Print final summary of pipeline execution"""
        self.logger.info("\n" + "="*80)
        self.logger.info("DEEPRAG PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Start Time: {self.pipeline_metrics['start_time']}")
        self.logger.info(f"End Time: {datetime.now().isoformat()}")
        self.logger.info(f"Total Questions: {self.pipeline_metrics['total_questions_processed']}")
        self.logger.info(f"Successful: {self.pipeline_metrics['successful_answers']}")
        self.logger.info(f"Failed: {self.pipeline_metrics['failed_answers']}")
        self.logger.info(f"Total Retrievals: {self.pipeline_metrics['total_retrievals']}")
        self.logger.info(f"Total Subqueries: {self.pipeline_metrics['total_subqueries']}")
        self.logger.info(f"Average Latency: {self.pipeline_metrics['average_latency_ms']:.0f}ms")
        self.logger.info("="*80)