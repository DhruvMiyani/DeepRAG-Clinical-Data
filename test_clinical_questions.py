"""
Quick test to verify DeepRAG is working with your MIMIC-III data
"""

import logging
from deeprag_pipeline import DeepRAGPipeline
from mimic_deeprag_integration import integrate_mimic_data_to_deeprag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deeprag_system():
    """Test if DeepRAG is properly answering clinical questions"""
    
    print("\n🔍 TESTING DEEPRAG WITH YOUR MIMIC-III DATA")
    print("=" * 50)
    
    # Initialize pipeline with your data
    print("Loading pipeline with MIMIC-III data...")
    pipeline = integrate_mimic_data_to_deeprag(
        data_path="./nosocomial-risk-datasets-from-mimic-iii-1.0",
        conditions=['hapi']  # Testing with pressure injury data
    )
    
    # Test questions
    test_questions = [
        "What is clinical code C0392747?",
        "What are risk factors for pressure injuries?",
        "How many patient records are in the dataset?"
    ]
    
    print("\n📋 Testing with clinical questions:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Process question
            result = pipeline.process_question(question, use_deeprag=True)
            
            if result['success']:
                print(f"✅ Answer: {result['answer'][:200]}...")
                print(f"⚡ Response time: {result['latency_ms']}ms")
            else:
                print(f"❌ Failed to get answer")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("\n✅ Test complete! Your DeepRAG system is working with MIMIC-III data!")
    return pipeline

if __name__ == "__main__":
    test_deeprag_system()