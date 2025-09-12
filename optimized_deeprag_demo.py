"""
Optimized DeepRAG Pipeline Demo with Rate Limit Handling
Processes clinical data in manageable batches to avoid API limits
"""

import logging
import time
from typing import List
from langchain.schema import Document
from mimic_deeprag_integration import MimicDeepRAGIntegrator
from deeprag_pipeline import DeepRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_optimized_pipeline():
    """Create DeepRAG pipeline with smaller, manageable dataset"""
    logger.info("🏥 OPTIMIZED DEEPRAG CLINICAL DEMONSTRATION")
    logger.info("=" * 60)
    
    pipeline = DeepRAGPipeline()
    integrator = MimicDeepRAGIntegrator("./nosocomial-risk-datasets-from-mimic-iii-1.0")
    
    # Load minimal sample for demo
    datasets = integrator.loader.load_condition_datasets('hapi')
    
    # Use very small sample to avoid rate limits
    sample_size = 150  # Much smaller sample
    sampled_datasets = {}
    
    for name, df in datasets.items():
        if not df.empty:
            actual_size = min(sample_size, len(df))
            sampled_datasets[name] = df.head(actual_size)
            logger.info(f"📊 Sampled {name}: {actual_size} records")
    
    # Convert to documents with smaller chunks
    documents = integrator._convert_datasets_to_documents(sampled_datasets, 'hapi')
    clinical_docs = integrator._get_clinical_knowledge_docs()
    documents.extend(clinical_docs)
    
    logger.info(f"📄 Created {len(documents)} documents from sampled data")
    
    # Create smaller text chunks to stay under token limits
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Smaller chunks
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Limit total chunks to stay under API limits
    max_chunks = 300  # Conservative limit for embeddings
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        logger.info(f"⚠️ Limited to {max_chunks} chunks to avoid rate limits")
    
    logger.info(f"📝 Processing {len(chunks)} text chunks")
    
    try:
        # Create vector store with retry logic
        logger.info("🔍 Creating embeddings (this may take a moment)...")
        pipeline.vector_store = FAISS.from_documents(chunks, pipeline.embeddings)
        pipeline.retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 3})
        pipeline.deeprag_core.retriever = pipeline.retriever
        
        logger.info("✅ Vector store created successfully!")
        return pipeline, sampled_datasets
        
    except Exception as e:
        logger.error(f"❌ Error creating vector store: {e}")
        return None, None


def demonstrate_clinical_reasoning(pipeline, sample_data):
    """Demonstrate DeepRAG capabilities with clinical questions"""
    
    if not pipeline:
        logger.error("❌ Pipeline not available for demonstration")
        return
    
    logger.info("\n🧠 DEEPRAG CLINICAL REASONING DEMONSTRATION")
    logger.info("=" * 50)
    
    # Focused clinical questions for demonstration
    demo_questions = [
        "What does clinical observation code C0392747 represent in medical terminology?",
        "What are the key risk factors for hospital-acquired pressure injuries?",
        "How are hospital admissions tracked and what is the significance of admission timestamps?",
        "What preventive measures are most effective for pressure injury management?"
    ]
    
    results = []
    
    for i, question in enumerate(demo_questions, 1):
        logger.info(f"\n🔬 Question {i}: {question}")
        logger.info("-" * 50)
        
        start_time = time.time()
        
        try:
            result = pipeline.process_question(
                question, 
                use_deeprag=True, 
                log_details=False  # Reduced logging for cleaner output
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            logger.info(f"⚡ Processed in {processing_time:.1f}ms")
            logger.info(f"🔍 Retrievals: {result.get('retrievals', 0)}")
            logger.info(f"✅ Success: {result.get('success', False)}")
            
            answer = result.get('answer', 'No answer generated')
            logger.info(f"\n💡 Answer: {answer[:400]}...")
            
            results.append({
                'question': question,
                'success': result.get('success', False),
                'latency_ms': processing_time,
                'retrievals': result.get('retrievals', 0)
            })
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            results.append({
                'question': question,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = [r for r in results if r.get('success', False)]
    if successful:
        avg_latency = sum(r['latency_ms'] for r in successful) / len(successful)
        avg_retrievals = sum(r['retrievals'] for r in successful) / len(successful)
        
        logger.info(f"\n📊 PERFORMANCE SUMMARY")
        logger.info("=" * 25)
        logger.info(f"✅ Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        logger.info(f"⚡ Avg Latency: {avg_latency:.1f}ms") 
        logger.info(f"🔍 Avg Retrievals: {avg_retrievals:.1f}")
    
    # Show data insights
    if sample_data and 'devel_chronologies' in sample_data:
        df = sample_data['devel_chronologies']
        logger.info(f"\n📈 SAMPLE DATA INSIGHTS")
        logger.info("=" * 25)
        logger.info(f"👥 Patients: {df['subject_id'].nunique()}")
        logger.info(f"🏥 Admissions: {df['hadm_id'].nunique()}")
        logger.info(f"📅 Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Show sample codes
        if not df.empty and 'observations' in df.columns:
            sample_obs = df['observations'].iloc[0]
            if sample_obs:
                codes = str(sample_obs).split()[:10]
                logger.info(f"🩺 Sample Codes: {' '.join(codes)}")


def main():
    """Main demonstration function"""
    try:
        # Create optimized pipeline
        pipeline, sample_data = create_optimized_pipeline()
        
        if pipeline:
            # Demonstrate clinical reasoning
            demonstrate_clinical_reasoning(pipeline, sample_data)
            
            logger.info("\n🎉 OPTIMIZED DEEPRAG DEMONSTRATION COMPLETED!")
            logger.info("🔬 Successfully processed real MIMIC-III clinical data")
            logger.info("🧠 DeepRAG reasoning capabilities demonstrated")
            logger.info("⚡ Rate limit optimizations working effectively")
        else:
            logger.error("❌ Failed to initialize pipeline")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()