"""
DeepRAG Training Components: Imitation Learning and Chain of Calibration
"""

import logging
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
# import torch
# import torch.nn as nn  
# import torch.nn.functional as F

from deeprag_core import DeepRAGCore, MDPState, MDPAction, BinaryTreeNode, DecisionType
from utils import FileManager, DataProcessor

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Training example for imitation learning"""
    question: str
    subqueries: List[str]
    decisions: List[str]  # retrieve or parametric
    intermediate_answers: List[str]
    final_answer: str
    retrieval_count: int


@dataclass
class PreferencePair:
    """Preference pair for chain of calibration"""
    state: MDPState
    subquery: str
    preferred_action: str  # retrieve or parametric
    rejected_action: str
    preferred_answer: str
    rejected_answer: str


class ImitationLearning:
    """Stage I: Imitation Learning from optimal trajectories"""
    
    def __init__(self, deeprag: DeepRAGCore):
        self.deeprag = deeprag
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_examples = []
        
    def synthesize_training_data(
        self,
        questions: List[Dict[str, str]],
        max_examples: int = 1000
    ) -> List[TrainingExample]:
        """Synthesize training data using binary tree search"""
        self.logger.info(f"Synthesizing training data for {len(questions)} questions")
        
        training_data = []
        
        for item in tqdm(questions[:max_examples], desc="Synthesizing data"):
            question = item['question']
            ground_truth = item.get('answer', '')
            
            try:
                # Perform binary tree search
                paths = self.deeprag.binary_tree_search(question, max_depth=5)
                
                if not paths:
                    self.logger.warning(f"No paths found for question: {question[:50]}...")
                    continue
                
                # Find optimal path
                optimal_path = self.deeprag.find_optimal_path(paths, ground_truth)
                
                if not optimal_path:
                    continue
                
                # Extract training example from optimal path
                example = self._extract_training_example(optimal_path)
                training_data.append(example)
                
                self.logger.debug(f"Created training example with {example.retrieval_count} retrievals")
                
            except Exception as e:
                self.logger.error(f"Error processing question: {e}")
                continue
        
        self.training_examples = training_data
        self.logger.info(f"Synthesized {len(training_data)} training examples")
        
        return training_data
    
    def _extract_training_example(self, path_node: BinaryTreeNode) -> TrainingExample:
        """Extract training example from optimal path"""
        trajectory = path_node.get_path()
        
        subqueries = []
        decisions = []
        intermediate_answers = []
        
        for state, action in trajectory:
            if action and action.subquery:
                subqueries.append(action.subquery)
                decisions.append(action.atomic_decision or DecisionType.PARAMETRIC.value)
                
                # Get the answer from state's last subquery
                if state.subqueries:
                    _, answer = state.subqueries[-1]
                    intermediate_answers.append(answer)
        
        return TrainingExample(
            question=path_node.state.question,
            subqueries=subqueries,
            decisions=decisions,
            intermediate_answers=intermediate_answers,
            final_answer=path_node.state.final_answer or "",
            retrieval_count=path_node.state.retrieval_count
        )
    
    def create_training_prompt(self, example: TrainingExample) -> str:
        """Create training prompt from example"""
        prompt = f"Question: {example.question}\n\n"
        
        for i, (subquery, decision, answer) in enumerate(
            zip(example.subqueries, example.decisions, example.intermediate_answers)
        ):
            prompt += f"Follow up: {subquery}\n"
            
            if decision == DecisionType.RETRIEVE.value:
                prompt += "Let's search the question in Wikipedia.\n"
                prompt += f"Context: [Retrieved information]\n"
            
            prompt += f"Intermediate answer: {answer}\n\n"
        
        prompt += f"So the final answer is: {example.final_answer}"
        
        return prompt
    
    def train_model(self, model, training_data: List[TrainingExample], epochs: int = 3):
        """Train model using imitation learning"""
        self.logger.info(f"Starting imitation learning for {epochs} epochs")
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_loss = 0.0
            
            for example in tqdm(training_data, desc=f"Epoch {epoch+1}/{epochs}"):
                # Create training prompt
                prompt = self.create_training_prompt(example)
                
                # Here you would implement actual model training
                # This is a placeholder for the training logic
                # In practice, you'd use the model's training API
                
                # Log example (for demonstration)
                if random.random() < 0.01:  # Log 1% of examples
                    self.logger.debug(f"Training on: {prompt[:200]}...")
            
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {total_loss/len(training_data):.4f}")
        
        self.logger.info("Imitation learning completed")
        return model


class ChainOfCalibration:
    """Stage II: Chain of Calibration for knowledge boundary awareness"""
    
    def __init__(self, deeprag: DeepRAGCore):
        self.deeprag = deeprag
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preference_pairs = []
    
    def synthesize_preference_data(
        self,
        questions: List[Dict[str, str]],
        imitation_model,
        max_examples: int = 500
    ) -> List[PreferencePair]:
        """Synthesize preference pairs for calibration"""
        self.logger.info(f"Synthesizing preference data for {len(questions)} questions")
        
        preference_data = []
        
        for item in tqdm(questions[:max_examples], desc="Creating preference pairs"):
            question = item['question']
            ground_truth = item.get('answer', '')
            
            try:
                # Get optimal path using trained model
                paths = self.deeprag.binary_tree_search(question, max_depth=5)
                
                if not paths:
                    continue
                
                optimal_path = self.deeprag.find_optimal_path(paths, ground_truth)
                
                if not optimal_path:
                    continue
                
                # Extract preference pairs from path
                pairs = self._extract_preference_pairs(optimal_path)
                preference_data.extend(pairs)
                
            except Exception as e:
                self.logger.error(f"Error creating preference pairs: {e}")
                continue
        
        self.preference_pairs = preference_data
        self.logger.info(f"Created {len(preference_data)} preference pairs")
        
        return preference_data
    
    def _extract_preference_pairs(self, optimal_path: BinaryTreeNode) -> List[PreferencePair]:
        """Extract preference pairs from optimal path"""
        pairs = []
        trajectory = optimal_path.get_path()
        
        for state, action in trajectory:
            if not action or not action.subquery:
                continue
            
            # Determine preferred and rejected actions
            if action.atomic_decision == DecisionType.RETRIEVE.value:
                preferred_action = DecisionType.RETRIEVE.value
                rejected_action = DecisionType.PARAMETRIC.value
            else:
                preferred_action = DecisionType.PARAMETRIC.value
                rejected_action = DecisionType.RETRIEVE.value
            
            # Get answers for both actions
            if state.subqueries:
                _, preferred_answer = state.subqueries[-1]
            else:
                preferred_answer = "[Answer from optimal path]"
            
            rejected_answer = "[Alternative answer]"
            
            pair = PreferencePair(
                state=MDPState(
                    question=state.question,
                    subqueries=state.subqueries[:-1] if state.subqueries else []
                ),
                subquery=action.subquery,
                preferred_action=preferred_action,
                rejected_action=rejected_action,
                preferred_answer=preferred_answer,
                rejected_answer=rejected_answer
            )
            
            pairs.append(pair)
        
        return pairs
    
    def compute_calibration_loss(
        self,
        model,
        preference_pair: PreferencePair,
        beta: float = 0.1
    ) -> float:
        """Compute chain of calibration loss"""
        # This is a simplified version of the loss computation
        # In practice, you'd implement the full DPO-style loss
        
        state_context = self._format_state_context(preference_pair.state)
        
        # Get model probabilities for both actions
        preferred_prompt = self._create_decision_prompt(
            state_context,
            preference_pair.subquery,
            preference_pair.preferred_action
        )
        
        rejected_prompt = self._create_decision_prompt(
            state_context,
            preference_pair.subquery,
            preference_pair.rejected_action
        )
        
        # Placeholder for actual probability computation
        # In practice, you'd get log probabilities from the model
        log_prob_preferred = -1.0  # Placeholder
        log_prob_rejected = -2.0   # Placeholder
        
        # DPO-style loss (simplified without torch)
        import math
        loss = -math.log(1 / (1 + math.exp(-beta * (log_prob_preferred - log_prob_rejected))))
        
        return loss
    
    def _format_state_context(self, state: MDPState) -> str:
        """Format state context for prompt"""
        context = f"Question: {state.question}\n\n"
        
        for i, (sq, ans) in enumerate(state.subqueries):
            context += f"Subquery {i+1}: {sq}\n"
            context += f"Answer {i+1}: {ans}\n\n"
        
        return context
    
    def _create_decision_prompt(self, context: str, subquery: str, decision: str) -> str:
        """Create prompt for decision"""
        prompt = context + f"\nNext subquery: {subquery}\n"
        
        if decision == DecisionType.RETRIEVE.value:
            prompt += "Decision: Let's search the question in Wikipedia.\n"
        else:
            prompt += "Decision: I can answer this with my knowledge.\n"
        
        return prompt
    
    def calibrate_model(
        self,
        model,
        preference_data: List[PreferencePair],
        epochs: int = 2,
        beta: float = 0.1
    ):
        """Calibrate model using preference learning"""
        self.logger.info(f"Starting chain of calibration for {epochs} epochs")
        
        for epoch in range(epochs):
            random.shuffle(preference_data)
            total_loss = 0.0
            
            for pair in tqdm(preference_data, desc=f"Calibration epoch {epoch+1}/{epochs}"):
                loss = self.compute_calibration_loss(model, pair, beta)
                total_loss += loss
                
                # Here you would implement actual model update
                # This is a placeholder
                
            avg_loss = total_loss / len(preference_data)
            self.logger.info(f"Calibration epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        self.logger.info("Chain of calibration completed")
        return model


class DeepRAGTrainer:
    """Main trainer combining both stages"""
    
    def __init__(self, deeprag: DeepRAGCore):
        self.deeprag = deeprag
        self.imitation_learning = ImitationLearning(deeprag)
        self.chain_calibration = ChainOfCalibration(deeprag)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_full_pipeline(
        self,
        training_questions: List[Dict[str, str]],
        calibration_questions: List[Dict[str, str]],
        model=None
    ):
        """Train full DeepRAG pipeline"""
        self.logger.info("Starting DeepRAG full training pipeline")
        
        # Stage I: Imitation Learning
        self.logger.info("=== Stage I: Imitation Learning ===")
        training_data = self.imitation_learning.synthesize_training_data(
            training_questions,
            max_examples=100  # Reduced for demo
        )
        
        if model:
            model = self.imitation_learning.train_model(model, training_data, epochs=2)
        
        # Save training data
        self._save_training_data(training_data, "imitation_training_data.json")
        
        # Stage II: Chain of Calibration
        self.logger.info("=== Stage II: Chain of Calibration ===")
        preference_data = self.chain_calibration.synthesize_preference_data(
            calibration_questions,
            model,
            max_examples=50  # Reduced for demo
        )
        
        if model:
            model = self.chain_calibration.calibrate_model(
                model,
                preference_data,
                epochs=1,
                beta=0.1
            )
        
        # Save preference data
        self._save_preference_data(preference_data, "calibration_preference_data.json")
        
        self.logger.info("DeepRAG training pipeline completed")
        return model
    
    def _save_training_data(self, data: List[TrainingExample], filepath: str):
        """Save training data to file"""
        serialized_data = []
        for example in data:
            serialized_data.append({
                'question': example.question,
                'subqueries': example.subqueries,
                'decisions': example.decisions,
                'intermediate_answers': example.intermediate_answers,
                'final_answer': example.final_answer,
                'retrieval_count': example.retrieval_count
            })
        
        FileManager.save_json(serialized_data, filepath)
        self.logger.info(f"Saved {len(data)} training examples to {filepath}")
    
    def _save_preference_data(self, data: List[PreferencePair], filepath: str):
        """Save preference data to file"""
        serialized_data = []
        for pair in data:
            serialized_data.append({
                'state': pair.state.to_dict(),
                'subquery': pair.subquery,
                'preferred_action': pair.preferred_action,
                'rejected_action': pair.rejected_action,
                'preferred_answer': pair.preferred_answer,
                'rejected_answer': pair.rejected_answer
            })
        
        FileManager.save_json(serialized_data, filepath)
        self.logger.info(f"Saved {len(data)} preference pairs to {filepath}")