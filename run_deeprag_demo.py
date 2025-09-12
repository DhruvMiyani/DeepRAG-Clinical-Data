"""
DeepRAG Demo Runner: Complete demonstration of DeepRAG pipeline
on clinical data with comprehensive logging
"""

import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

from deeprag_pipeline import DeepRAGPipeline
from utils import setup_logging

# Setup comprehensive logging
setup_logging(log_level="INFO", log_file="deeprag_demo.log")
logger = logging.getLogger(__name__)


def create_clinical_test_questions() -> List[str]:
    """Create test questions for clinical data demonstration"""
    return [
        "What are the risk factors for hospital-acquired pressure injuries?",
        "How is clinical observation code C0392747 related to pressure injuries?",
        "What is the significance of HADM_ID in tracking hospital-acquired conditions?",
        "What are the different stages of pressure injury classification?",
        "What prevention strategies are recommended for pressure injuries?",
        "How should clinical documentation be maintained for pressure injuries?",
        "What are the quality metrics used for hospital-acquired pressure injuries?",
        "What components are included in pressure injury prevention bundles?"
    ]


def create_training_questions() -> List[Dict[str, str]]:
    """Create training questions with answers for DeepRAG learning"""
    return [
        {
            "question": "What are the main risk factors for hospital-acquired pressure injuries?",
            "answer": "The main risk factors include immobility, poor nutrition, moisture, friction, and shear forces."
        },
        {
            "question": "What does clinical observation code C0392747 represent?",
            "answer": "C0392747 refers to pressure ulcer assessment including wound evaluation and risk scoring."
        },
        {
            "question": "How are pressure injuries classified?",
            "answer": "Pressure injuries are classified into stages 1-4 plus unstageable based on tissue damage depth."
        },
        {
            "question": "What is the Braden Scale used for?",
            "answer": "The Braden Scale assesses pressure injury risk with scores from 6-23, lower scores indicating higher risk."
        },
        {
            "question": "What constitutes a hospital-acquired condition?",
            "answer": "Conditions that develop 48+ hours after hospital admission are considered hospital-acquired."
        },
        {
            "question": "What are the key components of pressure injury prevention?",
            "answer": "Key components include risk assessment, skin assessment, repositioning, nutrition, and moisture management."
        }
    ]


def main():
    """Run complete DeepRAG demonstration"""
    logger.info("*" * 100)
    logger.info("DEEPRAG CLINICAL DATA DEMONSTRATION")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("*" * 100)
    
    try:
        # Initialize pipeline
        logger.info("\n=== PIPELINE INITIALIZATION ===")
        pipeline = DeepRAGPipeline()
        
        # Setup clinical knowledge base
        logger.info("\n=== CLINICAL KNOWLEDGE BASE SETUP ===")
        pipeline.setup_clinical_knowledge_base()
        
        # Get test questions
        test_questions = create_clinical_test_questions()
        training_questions = create_training_questions()
        
        logger.info(f"\nPrepared {len(test_questions)} test questions")
        logger.info(f"Prepared {len(training_questions)} training questions")
        
        # Run DeepRAG Training Pipeline (demonstration)
        logger.info("\n" + "="*80)
        logger.info("DEEPRAG TRAINING DEMONSTRATION")
        logger.info("="*80)
        
        # Note: This is a demonstration of the training process
        # In practice, you would train on larger datasets
        logger.info("Running DeepRAG training pipeline (demonstration mode)")
        
        if pipeline.trainer:
            # Split training data
            imitation_data = training_questions[:4]  # First 4 for imitation
            calibration_data = training_questions[2:]  # Last 4 for calibration
            
            logger.info(f"Using {len(imitation_data)} questions for imitation learning")
            logger.info(f"Using {len(calibration_data)} questions for chain of calibration")
            
            # Run training pipeline (demonstration)
            try:
                trained_model = pipeline.trainer.train_full_pipeline(
                    imitation_data,
                    calibration_data,
                    model=pipeline.llm  # Pass the LLM for demonstration
                )
                logger.info("DeepRAG training demonstration completed")
            except Exception as e:
                logger.error(f"Training demonstration encountered issues: {e}")
                logger.info("Proceeding with inference demonstration...")
        
        # Run Comparative Analysis
        logger.info("\n" + "="*80)
        logger.info("DEEPRAG vs STANDARD RAG COMPARISON")
        logger.info("="*80)
        
        # Select subset of questions for comparison
        comparison_questions = test_questions[:4]  # First 4 questions for demo
        
        logger.info(f"Running comparative analysis on {len(comparison_questions)} questions")
        comparison_results = pipeline.run_comparative_analysis(comparison_questions)
        
        # Individual Question Analysis
        logger.info("\n" + "="*80)
        logger.info("INDIVIDUAL QUESTION ANALYSIS")
        logger.info("="*80)
        
        # Process remaining questions individually
        individual_questions = test_questions[4:6]  # Next 2 questions
        individual_results = []
        
        for i, question in enumerate(individual_questions, 1):
            logger.info(f"\n--- Individual Question {i} ---")
            result = pipeline.process_question(question, use_deeprag=True, log_details=True)
            individual_results.append(result)
        
        # Compile all results
        all_results = {
            'comparison_analysis': comparison_results,
            'individual_analysis': individual_results,
            'metadata': {
                'total_questions_tested': len(test_questions[:6]),
                'deeprag_questions': len(comparison_questions) + len(individual_questions),
                'standard_rag_questions': len(comparison_questions),
                'test_type': 'clinical_data_demonstration'
            }
        }
        
        # Save results
        logger.info("\n=== SAVING RESULTS ===")
        pipeline.save_results(all_results, "deeprag_clinical_demo_results.json")
        
        # Print final summary
        pipeline.print_final_summary()
        
        # Additional Analysis Logging
        logger.info("\n" + "="*80)
        logger.info("DETAILED ANALYSIS SUMMARY")
        logger.info("="*80)
        
        # DeepRAG specific metrics
        deeprag_results = comparison_results['deeprag'] + individual_results
        deeprag_retrievals = [r['retrievals'] for r in deeprag_results]
        deeprag_subqueries = [len(r['subqueries']) for r in deeprag_results]
        
        logger.info(f"\nDeepRAG Performance:")
        logger.info(f"  Questions processed: {len(deeprag_results)}")
        logger.info(f"  Average retrievals per question: {sum(deeprag_retrievals)/len(deeprag_retrievals):.2f}")
        logger.info(f"  Average subqueries per question: {sum(deeprag_subqueries)/len(deeprag_subqueries):.2f}")
        logger.info(f"  Retrieval range: {min(deeprag_retrievals)} - {max(deeprag_retrievals)}")
        
        # Standard RAG metrics
        standard_results = comparison_results['standard']
        standard_retrievals = [r['retrievals'] for r in standard_results]
        
        logger.info(f"\nStandard RAG Performance:")
        logger.info(f"  Questions processed: {len(standard_results)}")
        logger.info(f"  Average retrievals per question: {sum(standard_retrievals)/len(standard_retrievals):.2f}")
        logger.info(f"  Retrieval range: {min(standard_retrievals)} - {max(standard_retrievals)}")
        
        # Efficiency comparison
        if standard_retrievals and deeprag_retrievals:
            avg_deeprag = sum(deeprag_retrievals[:len(standard_retrievals)]) / len(standard_retrievals)
            avg_standard = sum(standard_retrievals) / len(standard_retrievals)
            efficiency_gain = (1 - avg_deeprag/avg_standard) * 100 if avg_standard > 0 else 0
            
            logger.info(f"\nEfficiency Analysis:")
            logger.info(f"  DeepRAG avg retrievals: {avg_deeprag:.2f}")
            logger.info(f"  Standard RAG avg retrievals: {avg_standard:.2f}")
            logger.info(f"  Efficiency improvement: {efficiency_gain:.1f}%")
        
        # Log sample answers for quality assessment
        logger.info(f"\n=== SAMPLE ANSWERS QUALITY ASSESSMENT ===")
        for i, result in enumerate(deeprag_results[:2], 1):
            logger.info(f"\nSample {i}:")
            logger.info(f"Question: {result['question']}")
            logger.info(f"Answer: {result['answer'][:300]}...")
            logger.info(f"Retrievals: {result['retrievals']}")
            logger.info(f"Subqueries: {len(result['subqueries'])}")
        
        logger.info("\n" + "*" * 100)
        logger.info("DEEPRAG CLINICAL DATA DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info(f"Ended at: {datetime.now().isoformat()}")
        logger.info("*" * 100)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Run the demonstration
    results = main()
    
    print("\n" + "="*60)
    print("DeepRAG Clinical Data Demo Completed!")
    print("="*60)
    print(f"Results saved to: deeprag_clinical_demo_results.json")
    print(f"Detailed logs saved to: deeprag_demo.log")
    print(f"Pipeline logs saved to: deeprag_full_pipeline.log")
    print("="*60)