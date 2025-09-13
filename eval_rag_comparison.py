"""
RAG Evaluation Framework: DeepRAG vs Simple RAG Accuracy Comparison
Comprehensive evaluation system for clinical question answering
"""

import json
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import requests
from dataclasses import dataclass
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    ground_truth: str
    deeprag_answer: str
    simple_rag_answer: str
    deeprag_score: float
    simple_rag_score: float
    deeprag_latency: float
    simple_rag_latency: float
    deeprag_retrievals: int
    simple_rag_retrievals: int
    category: str
    difficulty: str

class ClinicalTestDataset:
    """Clinical question dataset with ground truth answers"""
    
    def __init__(self):
        self.test_questions = self._create_clinical_test_dataset()
    
    def _create_clinical_test_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive clinical test dataset"""
        
        return [
            # Clinical Code Interpretation (Easy)
            {
                "question": "What does clinical observation code C0392747 mean?",
                "ground_truth": "C0392747 refers to pressure injury or pressure ulcer, which is localized damage to skin and underlying tissue usually over bony prominences due to pressure or pressure in combination with shear.",
                "category": "clinical_codes",
                "difficulty": "easy",
                "keywords": ["pressure injury", "pressure ulcer", "skin damage", "bony prominence"],
                "expected_retrievals": 2
            },
            {
                "question": "What is the clinical meaning of code C0022116?",
                "ground_truth": "C0022116 represents acute kidney injury (AKI), which is a sudden decrease in kidney function characterized by rapid rise in serum creatinine and/or decrease in urine output.",
                "category": "clinical_codes", 
                "difficulty": "easy",
                "keywords": ["acute kidney injury", "AKI", "kidney function", "creatinine", "urine output"],
                "expected_retrievals": 2
            },
            {
                "question": "What does C0002871 indicate?",
                "ground_truth": "C0002871 indicates anemia, which is a condition characterized by decreased red blood cells or hemoglobin levels below normal ranges.",
                "category": "clinical_codes",
                "difficulty": "easy", 
                "keywords": ["anemia", "red blood cells", "hemoglobin", "blood"],
                "expected_retrievals": 2
            },
            
            # Risk Factor Analysis (Medium)
            {
                "question": "What are the main risk factors for hospital-acquired pressure injuries?",
                "ground_truth": "Main risk factors for HAPI include: immobility/prolonged bed rest, poor nutrition (low protein/calories), moisture from incontinence or perspiration, decreased sensory perception, friction and shear forces, advanced age, and low Braden Scale scores (<18).",
                "category": "risk_factors",
                "difficulty": "medium",
                "keywords": ["immobility", "nutrition", "moisture", "Braden Scale", "friction", "shear"],
                "expected_retrievals": 3
            },
            {
                "question": "What factors increase the risk of hospital-acquired acute kidney injury?",
                "ground_truth": "HAAKI risk factors include: pre-existing kidney disease, nephrotoxic medications, contrast agents, sepsis, hypotension, dehydration, advanced age, and certain surgical procedures.",
                "category": "risk_factors",
                "difficulty": "medium", 
                "keywords": ["nephrotoxic", "medications", "sepsis", "hypotension", "kidney disease"],
                "expected_retrievals": 3
            },
            
            # Assessment Tools (Medium)
            {
                "question": "How is the Braden Scale used for pressure injury risk assessment?",
                "ground_truth": "The Braden Scale assesses pressure injury risk using 6 factors: sensory perception, moisture, activity, mobility, nutrition, and friction/shear. Scores range from 6-23, with ≤12 indicating high risk, 13-14 moderate risk, 15-18 mild risk, and 19-23 low risk.",
                "category": "assessment_tools",
                "difficulty": "medium",
                "keywords": ["Braden Scale", "sensory perception", "mobility", "nutrition", "risk assessment"],
                "expected_retrievals": 3
            },
            {
                "question": "What are the KDIGO criteria for acute kidney injury?",
                "ground_truth": "KDIGO criteria define AKI by: serum creatinine increase ≥0.3 mg/dl within 48 hours, or creatinine increase ≥1.5x baseline within 7 days, or urine output <0.5 ml/kg/h for 6 hours. Staged as 1, 2, or 3 based on severity.",
                "category": "assessment_tools",
                "difficulty": "medium",
                "keywords": ["KDIGO", "creatinine", "urine output", "AKI staging"],
                "expected_retrievals": 3
            },
            
            # Complex Multi-Step Questions (Hard)
            {
                "question": "For a patient with mobility limitations and poor nutrition, what pressure injury prevention strategies should be implemented and how should they be monitored?",
                "ground_truth": "For high-risk patients: implement turning schedule every 2 hours, use pressure-relieving surfaces, optimize nutrition (protein 1.25-1.5 g/kg/day), manage moisture, conduct daily skin assessments, monitor Braden scores every 24-48 hours, document interventions, and educate staff on prevention protocols.",
                "category": "complex_care",
                "difficulty": "hard",
                "keywords": ["prevention", "turning schedule", "nutrition", "skin assessment", "monitoring"],
                "expected_retrievals": 4
            },
            {
                "question": "How should hospital-acquired conditions be identified, documented, and reported according to CMS guidelines?",
                "ground_truth": "HACs must be identified by timing (>48 hours post-admission), properly coded with ICD-10, documented with staging/severity, reported to CMS, tracked for quality metrics, and prevented through evidence-based protocols. Stage 3-4 pressure injuries are never events requiring root cause analysis.",
                "category": "complex_care", 
                "difficulty": "hard",
                "keywords": ["CMS", "documentation", "never events", "quality metrics", "ICD-10"],
                "expected_retrievals": 4
            },
            
            # Patient-Specific Scenarios (Hard)
            {
                "question": "What clinical timeline analysis would you perform for a patient developing pressure injury 5 days after admission?",
                "ground_truth": "Timeline analysis should include: admission skin assessment, daily Braden scores, turning schedule compliance, nutritional interventions, device placements, skin condition changes, intervention escalations, and determination of hospital-acquired status (>48 hours post-admission).",
                "category": "patient_analysis",
                "difficulty": "hard", 
                "keywords": ["timeline", "admission", "Braden scores", "hospital-acquired", "analysis"],
                "expected_retrievals": 4
            },
            
            # Edge Cases and Complex Medical Knowledge (Expert)
            {
                "question": "How do you differentiate between community-acquired and hospital-acquired anemia in patients with multiple comorbidities?",
                "ground_truth": "Differentiation requires: baseline hemoglobin levels at admission, timing of anemia onset (>48 hours for HAA), evaluation of blood loss sources (procedures, frequent draws), assessment of bone marrow suppression causes, nutritional factors, and underlying disease progression vs new hospital factors.",
                "category": "complex_diagnosis",
                "difficulty": "expert",
                "keywords": ["community-acquired", "hospital-acquired", "hemoglobin", "comorbidities", "timing"],
                "expected_retrievals": 5
            },
            
            # Quality Metrics and Outcomes (Expert)
            {
                "question": "How should pressure injury quality metrics be calculated and what benchmarks indicate good performance?",
                "ground_truth": "Quality metrics include: incidence rate (new cases per 1000 patient days), prevalence rate (point-in-time cases), healing rates, prevention compliance, and Braden assessment frequency. Benchmarks vary by unit type, with <5% incidence considered good performance in most acute care settings.",
                "category": "quality_metrics",
                "difficulty": "expert",
                "keywords": ["incidence rate", "prevalence", "benchmarks", "patient days", "performance"],
                "expected_retrievals": 5
            }
        ]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Get questions filtered by difficulty level"""
        return [q for q in self.test_questions if q["difficulty"] == difficulty]
    
    def get_questions_by_category(self, category: str) -> List[Dict]:
        """Get questions filtered by category"""
        return [q for q in self.test_questions if q["category"] == category]

class RAGEvaluator:
    """Evaluates and compares RAG approaches"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.dataset = ClinicalTestDataset()
    
    def query_rag_system(self, question: str, use_deeprag: bool = True) -> Dict[str, Any]:
        """Query the RAG system and return response with metrics"""
        
        url = f"{self.api_base_url}/ask"
        payload = {
            "question": question,
            "use_deeprag": use_deeprag,
            "log_details": True
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            end_time = time.time()
            
            return {
                "answer": result.get("answer", ""),
                "success": result.get("success", False),
                "latency_ms": result.get("latency_ms", (end_time - start_time) * 1000),
                "retrievals": result.get("retrievals", 0),
                "model_used": result.get("model_used", "unknown"),
                "method": "DeepRAG" if use_deeprag else "SimpleRAG"
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "success": False,
                "latency_ms": 0,
                "retrievals": 0,
                "model_used": "unknown",
                "method": "DeepRAG" if use_deeprag else "SimpleRAG"
            }
    
    def calculate_accuracy_score(self, answer: str, ground_truth: str, keywords: List[str]) -> float:
        """Calculate accuracy score based on content overlap and keyword presence"""
        
        if not answer or answer.startswith("Error:"):
            return 0.0
        
        # Normalize text for comparison
        answer_lower = answer.lower()
        ground_truth_lower = ground_truth.lower()
        
        scores = []
        
        # 1. Keyword Coverage (40% weight)
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0.5
        scores.append(keyword_score * 0.4)
        
        # 2. Content Similarity (30% weight) - Simple overlap measure
        answer_words = set(re.findall(r'\w+', answer_lower))
        truth_words = set(re.findall(r'\w+', ground_truth_lower))
        
        if truth_words:
            content_overlap = len(answer_words.intersection(truth_words)) / len(truth_words)
            scores.append(content_overlap * 0.3)
        else:
            scores.append(0)
        
        # 3. Length Appropriateness (10% weight)
        length_ratio = min(len(answer), len(ground_truth)) / max(len(answer), len(ground_truth))
        length_score = length_ratio if length_ratio > 0.3 else 0.3  # Minimum score for reasonable length
        scores.append(length_score * 0.1)
        
        # 4. Medical Accuracy Indicators (20% weight)
        medical_terms = ["injury", "pressure", "kidney", "anemia", "patient", "hospital", "clinical", "assessment"]
        medical_matches = sum(1 for term in medical_terms if term in answer_lower)
        medical_score = min(medical_matches / 3, 1.0)  # Cap at 1.0
        scores.append(medical_score * 0.2)
        
        total_score = sum(scores)
        return round(total_score, 3)
    
    def run_evaluation(self, questions_subset: List[Dict] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation comparing both approaches"""
        
        if questions_subset is None:
            questions_subset = self.dataset.test_questions
        
        logger.info(f"Starting evaluation with {len(questions_subset)} questions...")
        
        results = []
        
        for i, test_case in enumerate(questions_subset):
            logger.info(f"Evaluating question {i+1}/{len(questions_subset)}: {test_case['question'][:50]}...")
            
            # Test DeepRAG
            deeprag_result = self.query_rag_system(test_case["question"], use_deeprag=True)
            
            # Test Simple RAG  
            simple_rag_result = self.query_rag_system(test_case["question"], use_deeprag=False)
            
            # Calculate accuracy scores
            deeprag_score = self.calculate_accuracy_score(
                deeprag_result["answer"], 
                test_case["ground_truth"], 
                test_case["keywords"]
            )
            
            simple_rag_score = self.calculate_accuracy_score(
                simple_rag_result["answer"],
                test_case["ground_truth"], 
                test_case["keywords"]
            )
            
            # Create evaluation result
            result = EvaluationResult(
                question=test_case["question"],
                ground_truth=test_case["ground_truth"],
                deeprag_answer=deeprag_result["answer"],
                simple_rag_answer=simple_rag_result["answer"],
                deeprag_score=deeprag_score,
                simple_rag_score=simple_rag_score,
                deeprag_latency=deeprag_result["latency_ms"],
                simple_rag_latency=simple_rag_result["latency_ms"],
                deeprag_retrievals=deeprag_result["retrievals"],
                simple_rag_retrievals=simple_rag_result["retrievals"],
                category=test_case["category"],
                difficulty=test_case["difficulty"]
            )
            
            results.append(result)
            
            # Brief pause between questions
            time.sleep(1)
        
        logger.info("Evaluation completed!")
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return {"error": "No evaluation results provided"}
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in results:
            df_data.append({
                "question": result.question,
                "category": result.category,
                "difficulty": result.difficulty,
                "deeprag_score": result.deeprag_score,
                "simple_rag_score": result.simple_rag_score,
                "deeprag_latency": result.deeprag_latency,
                "simple_rag_latency": result.simple_rag_latency,
                "deeprag_retrievals": result.deeprag_retrievals,
                "simple_rag_retrievals": result.simple_rag_retrievals,
                "score_difference": result.deeprag_score - result.simple_rag_score
            })
        
        df = pd.DataFrame(df_data)
        
        # Overall Statistics
        overall_stats = {
            "total_questions": len(results),
            "deeprag_avg_score": df["deeprag_score"].mean(),
            "simple_rag_avg_score": df["simple_rag_score"].mean(),
            "deeprag_avg_latency": df["deeprag_latency"].mean(),
            "simple_rag_avg_latency": df["simple_rag_latency"].mean(),
            "deeprag_avg_retrievals": df["deeprag_retrievals"].mean(),
            "simple_rag_avg_retrievals": df["simple_rag_retrievals"].mean(),
            "deeprag_wins": len(df[df["score_difference"] > 0]),
            "simple_rag_wins": len(df[df["score_difference"] < 0]),
            "ties": len(df[df["score_difference"] == 0])
        }
        
        # Performance by Difficulty
        difficulty_stats = {}
        for difficulty in df["difficulty"].unique():
            subset = df[df["difficulty"] == difficulty]
            difficulty_stats[difficulty] = {
                "count": len(subset),
                "deeprag_avg_score": subset["deeprag_score"].mean(),
                "simple_rag_avg_score": subset["simple_rag_score"].mean(),
                "deeprag_wins": len(subset[subset["score_difference"] > 0]),
                "avg_score_difference": subset["score_difference"].mean()
            }
        
        # Performance by Category
        category_stats = {}
        for category in df["category"].unique():
            subset = df[df["category"] == category]
            category_stats[category] = {
                "count": len(subset),
                "deeprag_avg_score": subset["deeprag_score"].mean(),
                "simple_rag_avg_score": subset["simple_rag_score"].mean(),
                "deeprag_wins": len(subset[subset["score_difference"] > 0]),
                "avg_score_difference": subset["score_difference"].mean()
            }
        
        # Detailed Results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "question": result.question[:100] + "..." if len(result.question) > 100 else result.question,
                "category": result.category,
                "difficulty": result.difficulty,
                "deeprag_score": result.deeprag_score,
                "simple_rag_score": result.simple_rag_score,
                "winner": "DeepRAG" if result.deeprag_score > result.simple_rag_score else "SimpleRAG" if result.simple_rag_score > result.deeprag_score else "Tie",
                "score_difference": round(result.deeprag_score - result.simple_rag_score, 3),
                "deeprag_latency": round(result.deeprag_latency, 1),
                "simple_rag_latency": round(result.simple_rag_latency, 1)
            })
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "performance_by_difficulty": difficulty_stats,
            "performance_by_category": category_stats,
            "detailed_results": detailed_results,
            "summary": {
                "winner": "DeepRAG" if overall_stats["deeprag_avg_score"] > overall_stats["simple_rag_avg_score"] else "SimpleRAG",
                "improvement": abs(overall_stats["deeprag_avg_score"] - overall_stats["simple_rag_avg_score"]),
                "recommended_approach": "DeepRAG" if overall_stats["deeprag_avg_score"] > overall_stats["simple_rag_avg_score"] else "SimpleRAG"
            }
        }
    
    def save_results(self, results: List[EvaluationResult], report: Dict[str, Any], filename: str = None):
        """Save evaluation results and report to files"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_evaluation_{timestamp}"
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                "question": result.question,
                "ground_truth": result.ground_truth,
                "deeprag_answer": result.deeprag_answer,
                "simple_rag_answer": result.simple_rag_answer,
                "deeprag_score": result.deeprag_score,
                "simple_rag_score": result.simple_rag_score,
                "deeprag_latency": result.deeprag_latency,
                "simple_rag_latency": result.simple_rag_latency,
                "deeprag_retrievals": result.deeprag_retrievals,
                "simple_rag_retrievals": result.simple_rag_retrievals,
                "category": result.category,
                "difficulty": result.difficulty
            })
        
        # Save results
        with open(f"{filename}_detailed.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save report
        with open(f"{filename}_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results saved to {filename}_detailed.json and {filename}_report.json")

def main():
    """Main evaluation function"""
    
    print("🔬 RAG Evaluation Framework: DeepRAG vs Simple RAG")
    print("=" * 60)
    
    evaluator = RAGEvaluator()
    
    # Check if API is available
    try:
        response = requests.get(f"{evaluator.api_base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not available. Please start the server first.")
            return
    except:
        print("❌ Cannot connect to API server. Please start the server first.")
        return
    
    print("✅ API server is running")
    
    # Quick test with easy questions
    print("\n🧪 Running Quick Test (Easy Questions)...")
    easy_questions = evaluator.dataset.get_questions_by_difficulty("easy")
    quick_results = evaluator.run_evaluation(easy_questions)
    quick_report = evaluator.generate_report(quick_results)
    
    print(f"\n📊 Quick Test Results:")
    print(f"DeepRAG Average Score: {quick_report['overall_statistics']['deeprag_avg_score']:.3f}")
    print(f"SimpleRAG Average Score: {quick_report['overall_statistics']['simple_rag_avg_score']:.3f}")
    print(f"Winner: {quick_report['summary']['winner']}")
    
    # Ask user for full evaluation
    choice = input("\n🤔 Run full evaluation with all questions? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("\n🔄 Running Full Evaluation (All Questions)...")
        full_results = evaluator.run_evaluation()
        full_report = evaluator.generate_report(full_results)
        
        # Save results
        evaluator.save_results(full_results, full_report)
        
        # Print summary
        print(f"\n📈 Full Evaluation Results:")
        print(f"Total Questions: {full_report['overall_statistics']['total_questions']}")
        print(f"DeepRAG Average Score: {full_report['overall_statistics']['deeprag_avg_score']:.3f}")
        print(f"SimpleRAG Average Score: {full_report['overall_statistics']['simple_rag_avg_score']:.3f}")
        print(f"DeepRAG Wins: {full_report['overall_statistics']['deeprag_wins']}")
        print(f"SimpleRAG Wins: {full_report['overall_statistics']['simple_rag_wins']}")
        print(f"Recommended Approach: {full_report['summary']['recommended_approach']}")
        
        print(f"\n📄 Detailed results saved to evaluation files")
    
    else:
        print("✅ Quick evaluation completed!")

if __name__ == "__main__":
    main()