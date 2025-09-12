"""
Quick check to verify your DeepRAG system is working
"""

import os
import logging

print("\n🔍 CHECKING YOUR DEEPRAG SYSTEM")
print("=" * 50)

# 1. Check if data exists
data_path = "./nosocomial-risk-datasets-from-mimic-iii-1.0"
if os.path.exists(data_path):
    print("✅ MIMIC-III data found at:", data_path)
    
    # Check subdirectories
    conditions = ['hapi', 'haaki', 'haa']
    for condition in conditions:
        condition_path = os.path.join(data_path, condition)
        if os.path.exists(condition_path):
            files = os.listdir(condition_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"  ✅ {condition.upper()}: {len(csv_files)} CSV files")
else:
    print("❌ Data not found at:", data_path)

# 2. Check if environment is set up
print("\n📦 Checking environment:")
try:
    import openai
    print("✅ OpenAI library installed")
except:
    print("❌ OpenAI library not found")

try:
    import langchain
    print("✅ LangChain installed")
except:
    print("❌ LangChain not found")

try:
    from deeprag_pipeline import DeepRAGPipeline
    print("✅ DeepRAG pipeline module found")
except:
    print("❌ DeepRAG pipeline not found")

# 3. Check API key
from config import Config
if Config.OPENAI_API_KEY:
    print(f"✅ OpenAI API key configured (length: {len(Config.OPENAI_API_KEY)})")
else:
    print("❌ OpenAI API key not found")

# 4. Test basic functionality
print("\n🧪 Testing basic functionality:")
try:
    from datasets import NosocomialDataLoader
    loader = NosocomialDataLoader(data_path)
    
    # Quick load test
    datasets = loader.load_condition_datasets('hapi')
    if datasets:
        total_records = sum(len(df) for df in datasets.values())
        print(f"✅ Successfully loaded HAPI data: {total_records:,} total records")
        print(f"  - Training chronologies: {len(datasets.get('train_chronologies', []))} records")
        print(f"  - Development data: {len(datasets.get('devel_chronologies', []))} records")
        print(f"  - Test data: {len(datasets.get('test_chronologies', []))} records")
except Exception as e:
    print(f"❌ Error loading data: {e}")

print("\n" + "=" * 50)
print("🎯 SYSTEM STATUS:")
print("=" * 50)

# Overall status
all_good = all([
    os.path.exists(data_path),
    Config.OPENAI_API_KEY,
    'datasets' in locals()
])

if all_good:
    print("✅ YOUR DEEPRAG SYSTEM IS READY!")
    print("\nTo test with clinical questions, run:")
    print("  python3 test_mimic_integration.py")
    print("\nThis will:")
    print("  - Load a sample of your MIMIC-III data")
    print("  - Create embeddings for fast search")
    print("  - Answer clinical questions using DeepRAG")
else:
    print("⚠️ Some components need attention")
    print("Please check the errors above")

print("\n💡 Quick test commands:")
print("  - Small test: python3 test_mimic_integration.py")
print("  - Check data: python3 datasets.py")
print("  - Pipeline test: python3 deeprag_pipeline.py")