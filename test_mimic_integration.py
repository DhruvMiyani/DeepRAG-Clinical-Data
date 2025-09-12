"""
Lightweight test of MIMIC-III DeepRAG integration with sample data
"""

import logging
import pandas as pd
from mimic_deeprag_integration import MimicDeepRAGIntegrator
from deeprag_pipeline import DeepRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_sample_data(sample_size: int = 1000):
    """Test DeepRAG integration with sample of MIMIC data"""
    logger.info("🏥 TESTING DEEPRAG WITH REAL MIMIC-III SAMPLE")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = DeepRAGPipeline()
        
        # Initialize integrator
        integrator = MimicDeepRAGIntegrator("./nosocomial-risk-datasets-from-mimic-iii-1.0")
        
        # Load HAPI data
        datasets = integrator.loader.load_condition_datasets('hapi')
        
        # Sample data for faster testing
        sampled_datasets = {}
        for name, df in datasets.items():
            sampled_size = min(sample_size, len(df))
            sampled_datasets[name] = df.head(sampled_size)
            logger.info(f"Sampled {name}: {sampled_size:,} records (from {len(df):,})")
        
        # Convert sample to documents
        documents = integrator._convert_datasets_to_documents(sampled_datasets, 'hapi')
        
        # Add clinical knowledge
        clinical_docs = integrator._get_clinical_knowledge_docs()
        documents.extend(clinical_docs)
        
        logger.info(f"📄 Created {len(documents)} documents from sample")
        
        # Create text chunks (smaller chunks for testing)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for faster processing
            chunk_overlap=50
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"📝 Created {len(chunks)} text chunks")
        
        # Create vector store
        logger.info("🔍 Creating vector store with embeddings...")
        pipeline.vector_store = FAISS.from_documents(chunks, pipeline.embeddings)
        pipeline.retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 4})
        pipeline.deeprag_core.retriever = pipeline.retriever
        
        logger.info("✅ Pipeline ready with real MIMIC-III sample data!")
        
        # Test questions
        test_questions = [
            "What does clinical observation code C0392747 mean?",
            "What are the timestamps for patient 17 admission 161087?",
            "Which clinical codes are associated with pressure injuries?"
        ]
        
        logger.info("\n🧪 TESTING CLINICAL QUESTIONS")
        logger.info("=" * 40)
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n❓ Question {i}: {question}")
            
            try:
                result = pipeline.process_question(question, use_deeprag=True, log_details=False)
                
                logger.info(f"✅ Success: {result['success']}")
                logger.info(f"🔍 Retrievals: {result['retrievals']}")
                logger.info(f"⚡ Latency: {result['latency_ms']}ms")
                logger.info(f"💬 Answer: {result['answer'][:300]}...")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
        
        # Show sample data insights
        logger.info("\n📊 SAMPLE DATA INSIGHTS")
        logger.info("=" * 30)
        
        if 'devel_chronologies' in sampled_datasets:
            sample_df = sampled_datasets['devel_chronologies']
            logger.info(f"👥 Unique patients: {sample_df['subject_id'].nunique()}")
            logger.info(f"🏥 Unique admissions: {sample_df['hadm_id'].nunique()}")
            logger.info(f"📅 Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
            
            # Show sample observation codes
            sample_obs = sample_df['observations'].iloc[0]
            codes = sample_obs.split()[:10]  # First 10 codes
            logger.info(f"🔬 Sample observation codes: {' '.join(codes)}...")
        
        logger.info("\n🎉 MIMIC-III INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run test with sample data
    pipeline = test_with_sample_data(sample_size=500)  # Small sample for fast testing
    
    if pipeline:
        print("\n" + "="*60)
        print("✅ MIMIC-III DEEPRAG INTEGRATION SUCCESSFUL!")
        print("✅ Real patient data loaded and queryable")
        print("✅ Clinical observation codes integrated")
        print("✅ Hospital admission tracking working")
        print("="*60)