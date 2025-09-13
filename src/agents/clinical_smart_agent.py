"""
Clinical Smart Agent adapted from Microsoft DeepRAG
Integrates with existing MIMIC-III clinical data pipeline
"""

import base64
import inspect
import json
from logging import Logger
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

class ClinicalSmartAgent:
    """Smart agent for clinical RAG with tool function orchestration"""
    
    def __init__(
        self,
        logger: Logger,
        client: OpenAI,
        vector_retriever: Any,
        persona: str,
        model: str = "gpt-5",
        initial_message: str = "Hi, I'm your clinical research assistant. How can I help you with MIMIC-III data today?",
        max_run_per_question: int = 10,
        max_question_to_keep: int = 3,
        max_question_with_detail_hist: int = 1,
    ):
        self.logger = logger
        self.client = client
        self.vector_retriever = vector_retriever
        self.model = model
        self.persona = persona
        self.initial_message = initial_message
        self.max_run_per_question = max_run_per_question
        self.max_question_to_keep = max_question_to_keep
        self.max_question_with_detail_hist = max_question_with_detail_hist
        
        # Initialize conversation with persona
        self.conversation = [
            {"role": "system", "content": self.persona},
            {"role": "assistant", "content": self.initial_message}
        ]
        
        # Define clinical-specific tools
        self.functions_spec = self._get_clinical_tools()
        self.functions_list = {
            "search_clinical_records": self.search_clinical_records,
            "get_patient_timeline": self.get_patient_timeline,
            "identify_risk_factors": self.identify_risk_factors,
            "analyze_condition": self.analyze_condition
        }
    
    def _get_clinical_tools(self) -> List[ChatCompletionToolParam]:
        """Define clinical-specific tool functions"""
        return [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "search_clinical_records",
                    "description": "Search MIMIC-III clinical records using semantic search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about clinical conditions"
                            },
                            "condition_filter": {
                                "type": "string",
                                "enum": ["HAPI", "HAAKI", "HAA", "ALL"],
                                "description": "Filter by specific condition"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_patient_timeline",
                    "description": "Get temporal sequence of events for a patient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "integer",
                                "description": "Patient ID (subject_id)"
                            },
                            "admission_id": {
                                "type": "integer",
                                "description": "Hospital admission ID (hadm_id)"
                            }
                        },
                        "required": ["patient_id"]
                    }
                }
            ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "identify_risk_factors",
                    "description": "Identify risk factors for hospital-acquired conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "enum": ["HAPI", "HAAKI", "HAA"],
                                "description": "Condition to analyze"
                            }
                        },
                        "required": ["condition"]
                    }
                }
            ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "analyze_condition",
                    "description": "Analyze clinical observation codes and conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Clinical observation code (e.g., C0392747)"
                            }
                        },
                        "required": ["code"]
                    }
                }
            )
        ]
    
    def search_clinical_records(self, query: str, condition_filter: str = "ALL") -> List[Dict]:
        """Search clinical records using vector retrieval"""
        self.logger.info(f"Searching clinical records: {query}, filter: {condition_filter}")
        
        # Use existing vector retriever
        if self.vector_retriever:
            docs = self.vector_retriever.get_relevant_documents(query)
            
            # Filter by condition if specified
            if condition_filter != "ALL":
                docs = [d for d in docs if condition_filter.upper() in d.page_content.upper()]
            
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs[:4]]
        
        return []
    
    def get_patient_timeline(self, patient_id: int, admission_id: Optional[int] = None) -> Dict:
        """Get patient's clinical timeline"""
        self.logger.info(f"Getting timeline for patient {patient_id}")
        
        # Query for patient-specific data
        query = f"Patient ID: {patient_id}"
        if admission_id:
            query += f" Admission ID: {admission_id}"
        
        docs = self.search_clinical_records(query)
        
        # Parse and sort by timestamp if available
        timeline = []
        for doc in docs:
            content = doc["content"]
            # Extract timestamps and events from content
            lines = content.split("\n")
            for line in lines:
                if "Time:" in line:
                    timeline.append(line)
        
        return {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "events": timeline[:10]  # Limit to 10 events
        }
    
    def identify_risk_factors(self, condition: str) -> Dict:
        """Identify risk factors for a specific condition"""
        self.logger.info(f"Identifying risk factors for {condition}")
        
        # Knowledge base of risk factors
        risk_factors = {
            "HAPI": [
                "Immobility/prolonged bed rest",
                "Poor nutrition (low protein, low calorie)",
                "Moisture (incontinence, perspiration)",
                "Decreased sensory perception",
                "Friction and shear forces",
                "Advanced age",
                "Low Braden Scale score (<18)"
            ],
            "HAAKI": [
                "Pre-existing kidney disease",
                "Nephrotoxic medications",
                "Contrast agents",
                "Sepsis",
                "Hypotension",
                "Dehydration",
                "Advanced age"
            ],
            "HAA": [
                "Frequent blood draws",
                "Gastrointestinal bleeding",
                "Surgical procedures",
                "Bone marrow suppression",
                "Chronic disease",
                "Nutritional deficiencies"
            ]
        }
        
        return {
            "condition": condition,
            "risk_factors": risk_factors.get(condition, []),
            "assessment_tools": self._get_assessment_tools(condition)
        }
    
    def analyze_condition(self, code: str) -> Dict:
        """Analyze clinical observation codes"""
        self.logger.info(f"Analyzing clinical code: {code}")
        
        # Clinical code mappings
        code_mappings = {
            "C0392747": {
                "name": "Pressure Injury/Ulcer",
                "description": "Localized damage to skin and underlying tissue over bony prominence",
                "assessment": "Braden Scale, wound size/depth/stage evaluation",
                "icd10": "L89",
                "category": "Hospital-Acquired Pressure Injury (HAPI)"
            },
            "C0022116": {
                "name": "Acute Kidney Injury",
                "description": "Sudden decrease in kidney function",
                "assessment": "Serum creatinine, urine output, KDIGO criteria",
                "icd10": "N17",
                "category": "Hospital-Acquired Acute Kidney Injury (HAAKI)"
            },
            "C0002871": {
                "name": "Anemia",
                "description": "Decreased red blood cells or hemoglobin",
                "assessment": "Hemoglobin levels, hematocrit, CBC",
                "icd10": "D64.9",
                "category": "Hospital-Acquired Anemia (HAA)"
            }
        }
        
        return code_mappings.get(code, {"error": f"Unknown code: {code}"})
    
    def _get_assessment_tools(self, condition: str) -> List[str]:
        """Get assessment tools for a condition"""
        tools = {
            "HAPI": ["Braden Scale", "Norton Scale", "Waterlow Score"],
            "HAAKI": ["KDIGO Criteria", "RIFLE Criteria", "AKIN Classification"],
            "HAA": ["WHO Anemia Classification", "CBC Analysis"]
        }
        return tools.get(condition, [])
    
    def clean_up_history(self) -> None:
        """Clean up conversation history to manage context"""
        question_count = 0
        removal_indices = []
        
        for idx in range(len(self.conversation) - 1, 0, -1):
            message = dict(self.conversation[idx])
            
            if message.get("role") == "user":
                question_count += 1
            
            if question_count >= self.max_question_with_detail_hist and question_count < self.max_question_to_keep:
                if message.get("role") not in ["user", "assistant"] and len(message.get("content", [])) == 0:
                    removal_indices.append(idx)
            
            if question_count >= self.max_question_to_keep:
                removal_indices.append(idx)
        
        for index in removal_indices:
            del self.conversation[index]
    
    def run(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """Process user input and generate response"""
        if not user_input:
            return {
                "response": self.initial_message,
                "conversation": self.conversation,
                "success": True
            }
        
        # Add user input to conversation
        self.conversation.append({"role": "user", "content": user_input})
        self.clean_up_history()
        
        run_count = 0
        response_message = None
        
        while run_count < self.max_run_per_question:
            run_count += 1
            
            try:
                # Call OpenAI with tools
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation,
                    tools=self.functions_spec,
                    tool_choice='auto',
                    temperature=0.2,
                )
                
                response_message = response.choices[0].message
                
                if response_message.content is None:
                    response_message.content = ""
                
                # Handle tool calls
                tool_calls = response_message.tool_calls
                
                if tool_calls:
                    self.conversation.append(response_message)
                    self._process_tool_calls(tool_calls)
                    continue
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in agent run: {e}")
                response_message = ChatCompletionMessage(
                    role="assistant",
                    content=f"I encountered an error processing your request: {str(e)}"
                )
                break
        
        # Add final response to conversation
        self.conversation.append(response_message)
        
        return {
            "response": response_message.content,
            "conversation": self.conversation,
            "success": True,
            "tool_calls_made": run_count - 1
        }
    
    def _process_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]) -> None:
        """Process tool function calls"""
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            self.logger.debug(f"Calling function: {function_name}")
            
            if function_name not in self.functions_list:
                self.logger.error(f"Unknown function: {function_name}")
                self.conversation.pop()
                break
            
            try:
                function_args = json.loads(tool_call.function.arguments)
                function_to_call = self.functions_list[function_name]
                
                # Validate arguments
                if not self._check_args(function_to_call, function_args):
                    self.conversation.pop()
                    break
                
                # Execute function
                function_response = function_to_call(**function_args)
                
                # Add function response to conversation
                self.conversation.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
                
            except Exception as e:
                self.logger.error(f"Error calling {function_name}: {e}")
                self.conversation.pop()
                break
    
    def _check_args(self, function, args) -> bool:
        """Check if function has correct arguments"""
        sig = inspect.signature(function)
        params = sig.parameters
        
        # Check for extra arguments
        for name in args:
            if name not in params:
                return False
        
        # Check for required arguments
        for name, param in params.items():
            if param.default is param.empty and name not in args:
                return False
        
        return True