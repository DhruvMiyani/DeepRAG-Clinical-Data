"""
DeepRAG: Thinking to Retrieval Step by Step for Large Language Models
Implementation based on the paper by Guan et al., 2025
"""

import logging
import json
import time
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
import hashlib

import numpy as np
from tqdm import tqdm

from config import Config
from utils import FileManager, DataProcessor
from exceptions import RAGException

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deeprag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions in DeepRAG"""
    RETRIEVE = "retrieve"
    PARAMETRIC = "parametric"
    CONTINUE = "continue"
    TERMINATE = "terminate"


@dataclass
class MDPState:
    """Represents a state in the MDP formulation"""
    question: str
    subqueries: List[Tuple[str, str]] = field(default_factory=list)
    final_answer: Optional[str] = None
    retrieval_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def __hash__(self):
        """Create hash for state comparison"""
        state_str = f"{self.question}_{len(self.subqueries)}_{self.retrieval_count}"
        return int(hashlib.md5(state_str.encode()).hexdigest(), 16)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging"""
        return {
            'question': self.question,
            'subqueries': self.subqueries,
            'final_answer': self.final_answer,
            'retrieval_count': self.retrieval_count,
            'timestamp': self.timestamp
        }


@dataclass
class MDPAction:
    """Represents an action in the MDP"""
    termination_decision: str  # continue or terminate
    atomic_decision: Optional[str] = None  # retrieve or parametric
    subquery: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'termination': self.termination_decision,
            'atomic': self.atomic_decision,
            'subquery': self.subquery
        }


class BinaryTreeNode:
    """Node in the binary search tree for retrieval decisions"""
    def __init__(self, state: MDPState, action: Optional[MDPAction] = None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.reward = 0.0
        self.visited = False
        
    def add_child(self, child_state: MDPState, action: MDPAction):
        """Add a child node"""
        child = BinaryTreeNode(child_state, action, self)
        self.children.append(child)
        return child
    
    def get_path(self) -> List[Tuple[MDPState, MDPAction]]:
        """Get path from root to this node"""
        path = []
        node = self
        while node.parent:
            path.append((node.state, node.action))
            node = node.parent
        path.reverse()
        return path


class DeepRAGCore:
    """Core implementation of DeepRAG framework"""
    
    def __init__(self, llm, retriever, config: Config):
        """Initialize DeepRAG with LLM and retriever"""
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics tracking
        self.metrics = {
            'total_queries': 0,
            'total_retrievals': 0,
            'successful_answers': 0,
            'failed_answers': 0,
            'avg_retrieval_per_query': 0.0,
            'avg_subqueries_per_query': 0.0
        }
        
        self.logger.info("DeepRAG Core initialized successfully")
    
    def decompose_question(self, question: str, context: List[str] = None) -> str:
        """Decompose question into atomic subquery"""
        prompt = f"""You are decomposing a complex question into atomic subqueries.
        
Question: {question}

Previous context: {json.dumps(context) if context else "None"}

Generate the next atomic subquery that helps answer the main question.
If the question is fully answered, respond with "FINAL_ANSWER".

Subquery:"""
        
        response = self.llm.invoke(prompt)
        subquery = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        self.logger.debug(f"Generated subquery: {subquery}")
        return subquery
    
    def make_atomic_decision(self, subquery: str, state: MDPState) -> str:
        """Decide whether to retrieve or use parametric knowledge"""
        prompt = f"""Based on your internal knowledge, can you answer this query accurately?

Query: {subquery}

Previous context: {json.dumps([sq for sq, _ in state.subqueries[-3:]])}

Respond with only "RETRIEVE" if you need external information, or "PARAMETRIC" if you can answer with confidence.

Decision:"""
        
        response = self.llm.invoke(prompt)
        decision = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        if "RETRIEVE" in decision:
            return DecisionType.RETRIEVE.value
        else:
            return DecisionType.PARAMETRIC.value
    
    def generate_intermediate_answer(self, subquery: str, retrieved_docs: List[str] = None) -> str:
        """Generate intermediate answer for subquery"""
        if retrieved_docs:
            context = "\n".join(retrieved_docs[:3])  # Use top 3 docs
            prompt = f"""Answer the following query using the provided context.

Query: {subquery}

Context:
{context}

Answer:"""
        else:
            prompt = f"""Answer the following query using your knowledge.

Query: {subquery}

Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        self.logger.debug(f"Generated intermediate answer: {answer[:100]}...")
        return answer
    
    def generate_final_answer(self, state: MDPState) -> str:
        """Generate final answer from accumulated knowledge"""
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in state.subqueries])
        
        prompt = f"""Based on the following information, provide a comprehensive answer to the main question.

Main Question: {state.question}

Gathered Information:
{context}

Final Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        self.logger.info(f"Generated final answer for: {state.question[:50]}...")
        return answer
    
    def calculate_reward(self, state: MDPState, answer_correct: bool) -> float:
        """Calculate reward based on correctness and retrieval efficiency"""
        if not answer_correct:
            return -float('inf')
        
        # Reward = 1 / (1 + retrieval_count) to encourage minimal retrieval
        return 1.0 / (1.0 + state.retrieval_count)
    
    def binary_tree_search(self, question: str, max_depth: int = 5) -> List[BinaryTreeNode]:
        """Perform binary tree search for optimal retrieval path"""
        self.logger.info(f"Starting binary tree search for: {question[:50]}...")
        
        root_state = MDPState(question=question)
        root = BinaryTreeNode(root_state)
        
        # Priority queue for exploration (lower retrieval count = higher priority)
        pq = PriorityQueue()
        pq.put((0, id(root), root))
        
        complete_paths = []
        
        while not pq.empty() and len(complete_paths) < 10:
            retrieval_count, _, node = pq.get()
            
            if node.visited or len(node.state.subqueries) >= max_depth:
                continue
            
            node.visited = True
            
            # Generate next subquery
            subquery = self.decompose_question(
                node.state.question,
                [sq for sq, _ in node.state.subqueries]
            )
            
            if subquery == "FINAL_ANSWER":
                # Generate final answer and complete path
                final_answer = self.generate_final_answer(node.state)
                final_state = MDPState(
                    question=node.state.question,
                    subqueries=node.state.subqueries,
                    final_answer=final_answer,
                    retrieval_count=node.state.retrieval_count
                )
                final_node = node.add_child(
                    final_state,
                    MDPAction(termination_decision=DecisionType.TERMINATE.value)
                )
                complete_paths.append(final_node)
                self.logger.debug(f"Found complete path with {node.state.retrieval_count} retrievals")
                continue
            
            # Explore both retrieval options
            # Option 1: Use parametric knowledge
            param_answer = self.generate_intermediate_answer(subquery)
            param_state = MDPState(
                question=node.state.question,
                subqueries=node.state.subqueries + [(subquery, param_answer)],
                retrieval_count=node.state.retrieval_count
            )
            param_child = node.add_child(
                param_state,
                MDPAction(
                    termination_decision=DecisionType.CONTINUE.value,
                    atomic_decision=DecisionType.PARAMETRIC.value,
                    subquery=subquery
                )
            )
            pq.put((param_state.retrieval_count, id(param_child), param_child))
            
            # Option 2: Retrieve external knowledge
            retrieved_docs = self.retriever.similarity_search(subquery, k=3)
            retrieved_answer = self.generate_intermediate_answer(
                subquery,
                [doc.page_content for doc in retrieved_docs]
            )
            retrieve_state = MDPState(
                question=node.state.question,
                subqueries=node.state.subqueries + [(subquery, retrieved_answer)],
                retrieval_count=node.state.retrieval_count + 1
            )
            retrieve_child = node.add_child(
                retrieve_state,
                MDPAction(
                    termination_decision=DecisionType.CONTINUE.value,
                    atomic_decision=DecisionType.RETRIEVE.value,
                    subquery=subquery
                )
            )
            pq.put((retrieve_state.retrieval_count, id(retrieve_child), retrieve_child))
        
        self.logger.info(f"Binary tree search completed. Found {len(complete_paths)} paths")
        return complete_paths
    
    def find_optimal_path(self, paths: List[BinaryTreeNode], ground_truth: str = None) -> BinaryTreeNode:
        """Find optimal path with minimal retrieval cost"""
        if not paths:
            return None
        
        # If ground truth provided, filter for correct answers
        if ground_truth:
            correct_paths = []
            for path in paths:
                if self.evaluate_answer(path.state.final_answer, ground_truth):
                    correct_paths.append(path)
            paths = correct_paths if correct_paths else paths
        
        # Select path with minimal retrieval count
        optimal_path = min(paths, key=lambda p: p.state.retrieval_count)
        
        self.logger.info(f"Found optimal path with {optimal_path.state.retrieval_count} retrievals")
        return optimal_path
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate if predicted answer matches ground truth"""
        # Simple evaluation - can be enhanced with better metrics
        pred_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # Check for exact match or containment
        return truth_lower in pred_lower or pred_lower in truth_lower
    
    def log_metrics(self):
        """Log current metrics"""
        self.logger.info("=== DeepRAG Metrics ===")
        for key, value in self.metrics.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=====================")
    
    def save_trajectory(self, trajectory: List[Tuple[MDPState, MDPAction]], filepath: str):
        """Save trajectory to file for analysis"""
        trajectory_data = []
        for state, action in trajectory:
            trajectory_data.append({
                'state': state.to_dict(),
                'action': action.to_dict() if action else None
            })
        
        FileManager.save_json(trajectory_data, filepath)
        self.logger.info(f"Trajectory saved to {filepath}")