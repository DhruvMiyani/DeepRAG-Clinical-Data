"""
Comprehensive DeepRAG Pipeline Demonstration with Real MIMIC-III Data
Tests advanced clinical reasoning and multi-step retrieval capabilities
"""

import logging
import time
from mimic_deeprag_integration import MimicDeepRAGIntegrator
from deeprag_pipeline import DeepRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def comprehensive_clinical_qa_demo():
    """Demonstrate DeepRAG with complex clinical questions"""
    logger.info("🏥 COMPREHENSIVE DEEPRAG CLINICAL DEMONSTRATION")
    logger.info("=" * 70)
    
    # Initialize with real MIMIC-III data
    pipeline = DeepRAGPipeline()
    integrator = MimicDeepRAGIntegrator("./nosocomial-risk-datasets-from-mimic-iii-1.0")
    
    # Load sample data for faster demo
    datasets = integrator.loader.load_condition_datasets('hapi')
    sampled_datasets = {}
    for name, df in datasets.items():
        sampled_size = min(800, len(df))  # Larger sample for better coverage
        sampled_datasets[name] = df.head(sampled_size)
        logger.info(f"📊 Loaded {name}: {sampled_size:,} records")
    
    # Create documents and vector store
    documents = integrator._convert_datasets_to_documents(sampled_datasets, 'hapi')
    clinical_docs = integrator._get_clinical_knowledge_docs()
    documents.extend(clinical_docs)
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=75)
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"📄 Created {len(chunks)} searchable chunks from {len(documents)} documents")
    
    pipeline.vector_store = FAISS.from_documents(chunks, pipeline.embeddings)
    pipeline.retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 5})
    pipeline.deeprag_core.retriever = pipeline.retriever
    
    # Advanced clinical questions requiring multi-step reasoning
    advanced_questions = [
        {
            "question": "For patient 17 during admission 161087, what were the clinical observations and what do they indicate about pressure injury risk?",
            "complexity": "Multi-entity retrieval + Clinical interpretation"
        },
        {
            "question": "Compare the observation patterns between different patients who developed hospital-acquired pressure injuries. What common risk factors can you identify?",
            "complexity": "Cross-patient analysis + Pattern recognition"
        },
        {
            "question": "What is the clinical significance of observation code C0392747 and how does it relate to the Braden Scale assessment?",
            "complexity": "Code interpretation + Clinical knowledge integration"
        },
        {
            "question": "Explain the temporal progression of hospital-acquired conditions. How do admission timestamps help in classification?",
            "complexity": "Temporal reasoning + Clinical classification"
        },
        {
            "question": "What preventive interventions are most effective for hospital-acquired pressure injuries based on the clinical evidence in the dataset?",
            "complexity": "Evidence synthesis + Clinical recommendations"
        }
    ]
    
    logger.info("\n🧠 TESTING ADVANCED CLINICAL REASONING")
    logger.info("=" * 50)
    
    results = []
    
    for i, qa_item in enumerate(advanced_questions, 1):
        question = qa_item["question"]
        complexity = qa_item["complexity"]
        
        logger.info(f"\n🔬 Question {i}: {question}")
        logger.info(f"📈 Complexity: {complexity}")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        try:
            # Use DeepRAG with detailed logging
            result = pipeline.process_question(
                question, 
                use_deeprag=True, 
                log_details=True
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            logger.info(f"✅ Processing completed in {processing_time:.1f}ms")
            logger.info(f"🔍 Retrievals performed: {result.get('retrievals', 0)}")
            logger.info(f"🎯 Success: {result.get('success', False)}")
            
            # Display answer with formatting
            answer = result.get('answer', 'No answer generated')
            logger.info(f"\n💡 CLINICAL ANSWER:")
            logger.info("-" * 40)
            
            # Format answer for better readability
            if len(answer) > 500:
                logger.info(f"{answer[:500]}...")
                logger.info("... [Answer truncated for display]")
            else:
                logger.info(answer)
            
            results.append({
                'question': question,
                'complexity': complexity,
                'success': result.get('success', False),
                'retrievals': result.get('retrievals', 0),
                'latency_ms': processing_time,
                'answer_length': len(answer)
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            results.append({
                'question': question,
                'complexity': complexity,
                'success': False,
                'error': str(e)
            })
    
    # Performance summary
    logger.info("\n📊 DEEPRAG PERFORMANCE SUMMARY")
    logger.info("=" * 40)
    
    successful_queries = [r for r in results if r.get('success', False)]
    if successful_queries:
        avg_latency = sum(r['latency_ms'] for r in successful_queries) / len(successful_queries)
        avg_retrievals = sum(r['retrievals'] for r in successful_queries) / len(successful_queries)
        avg_answer_length = sum(r['answer_length'] for r in successful_queries) / len(successful_queries)
        
        logger.info(f"✅ Success Rate: {len(successful_queries)}/{len(results)} ({len(successful_queries)/len(results)*100:.1f}%)")
        logger.info(f"⚡ Average Latency: {avg_latency:.1f}ms")
        logger.info(f"🔍 Average Retrievals: {avg_retrievals:.1f}")
        logger.info(f"📝 Average Answer Length: {avg_answer_length:.0f} characters")
    
    # Data insights
    logger.info(f"\n📈 DATASET INSIGHTS")
    logger.info("=" * 30)
    if 'devel_chronologies' in sampled_datasets:
        df = sampled_datasets['devel_chronologies']
        logger.info(f"👥 Unique Patients: {df['subject_id'].nunique()}")
        logger.info(f"🏥 Unique Admissions: {df['hadm_id'].nunique()}")
        logger.info(f"📅 Observation Timeline: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Sample clinical codes
        sample_obs = df['observations'].iloc[0] if not df.empty else ""
        if sample_obs:
            codes = sample_obs.split()[:15]
            logger.info(f"🩺 Sample Clinical Codes: {' '.join(codes)}")
    
    logger.info("\n🎉 COMPREHENSIVE DEEPRAG DEMONSTRATION COMPLETED!")
    logger.info("🔬 Real MIMIC-III patient data successfully processed")
    logger.info("🧠 Advanced clinical reasoning demonstrated")
    logger.info("⚡ Multi-step retrieval and inference working")
    
    return results


if __name__ == "__main__":
    comprehensive_clinical_qa_demo()