"""
Quick RAG Comparison Demo
Shows how to evaluate DeepRAG vs Simple RAG accuracy
"""

import requests
import time

def test_rag_comparison():
    """Quick demo of RAG comparison"""
    
    api_url = "http://localhost:8001/ask"
    
    # Test question
    question = "What does clinical observation code C0392747 mean?"
    print(f"Question: {question}")
    print("=" * 60)
    
    # Test DeepRAG
    print("🧠 Testing DeepRAG Smart Agent...")
    start_time = time.time()
    
    deeprag_response = requests.post(api_url, json={
        "question": question,
        "use_deeprag": True,
        "log_details": True
    })
    
    deeprag_latency = (time.time() - start_time) * 1000
    deeprag_result = deeprag_response.json()
    
    print(f"✅ DeepRAG Answer: {deeprag_result['answer']}")
    print(f"   Latency: {deeprag_latency:.1f}ms")
    print(f"   Retrievals: {deeprag_result['retrievals']}")
    print(f"   Method: DeepRAG-SmartAgent")
    print()
    
    # Test Simple RAG
    print("🔍 Testing Simple RAG...")
    start_time = time.time()
    
    simple_response = requests.post(api_url, json={
        "question": question,
        "use_deeprag": False,
        "log_details": True  
    })
    
    simple_latency = (time.time() - start_time) * 1000
    simple_result = simple_response.json()
    
    print(f"✅ Simple RAG Answer: {simple_result['answer']}")
    print(f"   Latency: {simple_latency:.1f}ms")
    print(f"   Retrievals: {simple_result['retrievals']}")
    print(f"   Method: Basic RAG")
    print()
    
    # Simple comparison
    print("📊 Quick Comparison:")
    print(f"   DeepRAG Latency: {deeprag_latency:.1f}ms")
    print(f"   Simple RAG Latency: {simple_latency:.1f}ms")
    print(f"   Winner (speed): {'Simple RAG' if simple_latency < deeprag_latency else 'DeepRAG'}")
    
    # Content analysis
    deeprag_keywords = ["pressure", "injury", "ulcer", "skin", "C0392747"]
    simple_keywords = ["pressure", "injury", "ulcer", "skin", "C0392747"]
    
    deeprag_matches = sum(1 for k in deeprag_keywords if k.lower() in deeprag_result['answer'].lower())
    simple_matches = sum(1 for k in simple_keywords if k.lower() in simple_result['answer'].lower())
    
    print(f"   DeepRAG Keywords: {deeprag_matches}/{len(deeprag_keywords)}")
    print(f"   Simple RAG Keywords: {simple_matches}/{len(simple_keywords)}")
    print(f"   Winner (content): {'DeepRAG' if deeprag_matches > simple_matches else 'Simple RAG' if simple_matches > deeprag_matches else 'Tie'}")

if __name__ == "__main__":
    test_rag_comparison()