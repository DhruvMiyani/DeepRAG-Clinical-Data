# 🔬 RAG Evaluation Guide: DeepRAG vs Simple RAG Accuracy Comparison

## 📋 Overview

This guide provides a comprehensive framework to evaluate and compare the accuracy of **DeepRAG Smart Agent** vs **Simple RAG** approaches for clinical question answering using MIMIC-III data.

## 🎯 Key Findings from Initial Testing

### Quick Test Results (3 Easy Questions):
- **DeepRAG Average Score**: 0.344
- **SimpleRAG Average Score**: 0.358  
- **Winner**: SimpleRAG (marginally)
- **Speed Winner**: DeepRAG (14.9s vs 20.5s)

### Individual Question Performance:
| Question | DeepRAG | SimpleRAG | Winner | 
|----------|---------|-----------|--------|
| C0392747 meaning | ✅ 5/5 keywords | ✅ 5/5 keywords | Tie |
| Speed comparison | 14.9s | 20.5s | DeepRAG |

## 🏗️ Evaluation Framework Architecture

### Components:

1. **ClinicalTestDataset**: 12 curated questions with ground truth
2. **RAGEvaluator**: Automated testing system
3. **Accuracy Scoring**: Multi-dimensional evaluation
4. **Performance Metrics**: Speed, retrievals, content quality

## 📊 Test Dataset Categories

### 1. Clinical Code Interpretation (Easy - 3 questions)
- C0392747 (Pressure Injury)
- C0022116 (Acute Kidney Injury)  
- C0002871 (Anemia)

### 2. Risk Factor Analysis (Medium - 2 questions)
- HAPI risk factors
- HAAKI risk factors

### 3. Assessment Tools (Medium - 2 questions)
- Braden Scale usage
- KDIGO criteria

### 4. Complex Multi-Step Questions (Hard - 2 questions)
- Prevention strategies
- HAC identification/reporting

### 5. Patient-Specific Scenarios (Hard - 1 question)
- Timeline analysis

### 6. Expert-Level Questions (Expert - 2 questions)
- Community vs hospital-acquired differentiation
- Quality metrics calculation

## 🧮 Scoring Methodology

### Accuracy Score Components (0.0 - 1.0):

1. **Keyword Coverage (40% weight)**
   - Matches clinical keywords from ground truth
   - Example: ["pressure", "injury", "ulcer", "skin"]

2. **Content Similarity (30% weight)**
   - Word overlap between answer and ground truth
   - Measures factual accuracy

3. **Length Appropriateness (10% weight)**
   - Penalizes extremely short or long answers
   - Ensures comprehensive responses

4. **Medical Accuracy Indicators (20% weight)**
   - Presence of medical terminology
   - Clinical relevance assessment

## 🚀 How to Run Evaluations

### Method 1: Quick Demo (Single Question)
```bash
python3 quick_rag_demo.py
```
**Output:**
- Side-by-side comparison
- Speed analysis
- Keyword matching
- Winner determination

### Method 2: Comprehensive Evaluation (All Questions)
```bash
python3 eval_rag_comparison.py
```
**Features:**
- Tests all 12 questions
- Saves detailed JSON reports
- Statistical analysis by difficulty/category
- Performance benchmarking

### Method 3: Custom Evaluation
```python
from eval_rag_comparison import RAGEvaluator, ClinicalTestDataset

evaluator = RAGEvaluator()
dataset = ClinicalTestDataset()

# Test specific difficulty level
easy_questions = dataset.get_questions_by_difficulty("easy")
results = evaluator.run_evaluation(easy_questions)
report = evaluator.generate_report(results)
```

## 📈 Interpretation of Results

### Score Ranges:
- **0.8 - 1.0**: Excellent (Comprehensive, accurate answer)
- **0.6 - 0.8**: Good (Most key points covered)
- **0.4 - 0.6**: Fair (Some accuracy, missing details)
- **0.2 - 0.4**: Poor (Limited accuracy)
- **0.0 - 0.2**: Very Poor (Incorrect or no answer)

### Performance Indicators:

#### **DeepRAG Advantages:**
- ✅ **Faster responses** (14.9s vs 20.5s)
- ✅ **Tool-based reasoning** (search, timeline, risk analysis)
- ✅ **Multi-turn conversation** capability
- ✅ **Clinical-specific persona** and context
- ✅ **Structured clinical knowledge** integration

#### **SimpleRAG Advantages:**
- ✅ **Slightly higher accuracy** (0.358 vs 0.344)
- ✅ **More predictable** responses
- ✅ **Lower complexity** 
- ✅ **Easier debugging**
- ✅ **Resource efficient**

## 🎯 When to Use Which Approach

### Use **DeepRAG** for:
- Complex clinical workflows
- Multi-step reasoning tasks
- Patient timeline analysis
- Risk factor identification
- Production systems with scale
- Interactive clinical assistants

### Use **SimpleRAG** for:
- Simple fact lookup
- Single-answer questions
- Prototype development
- Limited computational resources
- Quick clinical code interpretation

## 🔧 Customizing the Evaluation

### Adding New Test Questions:
```python
# In ClinicalTestDataset._create_clinical_test_dataset()
{
    "question": "Your custom clinical question?",
    "ground_truth": "Expected accurate answer",
    "category": "custom_category",
    "difficulty": "medium",
    "keywords": ["key", "clinical", "terms"],
    "expected_retrievals": 3
}
```

### Modifying Scoring Weights:
```python
# In RAGEvaluator.calculate_accuracy_score()
scores.append(keyword_score * 0.5)  # Increase keyword weight
scores.append(content_overlap * 0.2)  # Decrease content weight
```

### Testing Different Models:
Update your `.env` file:
```bash
DEFAULT_MODEL=gpt-4  # or gpt-3.5-turbo
ENABLE_DEEPRAG=false  # Test SimpleRAG only
```

## 📊 Sample Evaluation Report Structure

```json
{
  "evaluation_timestamp": "2025-09-13T14:30:00",
  "overall_statistics": {
    "total_questions": 12,
    "deeprag_avg_score": 0.745,
    "simple_rag_avg_score": 0.682,
    "deeprag_wins": 8,
    "simple_rag_wins": 3,
    "ties": 1
  },
  "performance_by_difficulty": {
    "easy": {"deeprag_avg_score": 0.823, "simple_rag_avg_score": 0.789},
    "medium": {"deeprag_avg_score": 0.712, "simple_rag_avg_score": 0.645},
    "hard": {"deeprag_avg_score": 0.634, "simple_rag_avg_score": 0.578}
  },
  "summary": {
    "winner": "DeepRAG",
    "improvement": 0.063,
    "recommended_approach": "DeepRAG"
  }
}
```

## 🎨 Visualization Ideas

### Create charts showing:
1. **Accuracy by Difficulty Level**
2. **Response Time Comparison**
3. **Retrieval Efficiency**  
4. **Category Performance Heatmap**
5. **Score Distribution Histograms**

### Example with matplotlib:
```python
import matplotlib.pyplot as plt
import pandas as pd

# Load evaluation results
df = pd.read_json("rag_evaluation_20250913_143000_report.json")

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(["DeepRAG", "SimpleRAG"], 
        [report['overall_statistics']['deeprag_avg_score'],
         report['overall_statistics']['simple_rag_avg_score']])
plt.ylabel("Average Accuracy Score")
plt.title("RAG Approach Comparison")
plt.show()
```

## 🔍 Debugging Poor Performance

### If DeepRAG Scores Low:
1. Check Smart Agent tool calls in logs
2. Verify clinical ontology coverage
3. Ensure proper persona configuration
4. Review tool function implementations

### If SimpleRAG Scores Low:
1. Check vector store quality
2. Verify chunk size/overlap settings
3. Review retrieval parameters (k value)
4. Analyze document preprocessing

## 🚀 Advanced Evaluation Features

### 1. Human Evaluation Integration:
```python
def human_evaluation_score(answer: str) -> float:
    score = float(input(f"Rate this answer (0-1): {answer[:100]}..."))
    return score
```

### 2. Medical Expert Validation:
- Export results to clinical review format
- Include confidence intervals
- Add statistical significance testing

### 3. Continuous Evaluation:
- Schedule regular evaluations
- Track performance over time
- Monitor model drift

## 📚 Files in Evaluation Framework

### Core Files:
- `eval_rag_comparison.py` - Main evaluation framework
- `quick_rag_demo.py` - Simple demonstration
- `RAG_EVALUATION_GUIDE.md` - This documentation

### Generated Output Files:
- `rag_evaluation_YYYYMMDD_HHMMSS_detailed.json` - Detailed results
- `rag_evaluation_YYYYMMDD_HHMMSS_report.json` - Summary report

## 🎯 Best Practices

### 1. **Consistent Testing Environment**
- Same API server configuration
- Identical MIMIC-III dataset
- Consistent model parameters

### 2. **Statistical Rigor**
- Run multiple evaluation rounds
- Calculate confidence intervals
- Use appropriate sample sizes

### 3. **Clinical Relevance**
- Include domain expert review
- Test on realistic clinical scenarios
- Validate against actual use cases

### 4. **Performance Monitoring**
- Track latency trends
- Monitor accuracy over time
- Alert on significant degradation

## 🔮 Future Enhancements

### Planned Features:
1. **Interactive Evaluation Dashboard**
2. **A/B Testing Framework**
3. **Real-time Performance Monitoring**
4. **Clinical Expert Review Interface**
5. **Automated Model Selection**
6. **Cross-validation Testing**
7. **Bias Detection Analysis**

## 📞 Usage Examples

### Basic Usage:
```bash
# Quick test
python3 quick_rag_demo.py

# Full evaluation
python3 eval_rag_comparison.py

# View results
cat rag_evaluation_*_report.json | jq '.summary'
```

### Advanced Usage:
```python
# Custom evaluation
evaluator = RAGEvaluator()
expert_questions = evaluator.dataset.get_questions_by_difficulty("expert")
results = evaluator.run_evaluation(expert_questions)

# Save with custom filename
evaluator.save_results(results, report, "expert_level_evaluation")
```

This evaluation framework provides a robust foundation for comparing RAG approaches and making data-driven decisions about your clinical AI system! 🎉