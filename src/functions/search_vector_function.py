"""
Search Vector Function adapted from Microsoft DeepRAG
Integrates with existing clinical data pipeline and Azure AI Search
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class SearchResult:
    """Represents a search result from vector search"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class SearchVectorFunction:
    """Clinical search function adapted from Microsoft DeepRAG architecture"""
    
    def __init__(
        self,
        logger: logging.Logger,
        vector_retriever: Any,
        use_azure_search: bool = False,
        azure_search_client: Optional[Any] = None
    ):
        self.logger = logger
        self.vector_retriever = vector_retriever
        self.use_azure_search = use_azure_search
        self.azure_search_client = azure_search_client
        
    def search(
        self,
        search_query: str,
        condition_filter: str = "ALL",
        max_results: int = 6
    ) -> List[SearchResult]:
        """
        Perform semantic search for clinical content
        
        Args:
            search_query: Natural language query
            condition_filter: Filter by condition (HAPI, HAAKI, HAA, ALL)
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        self.logger.info(f"Searching for: '{search_query}' with filter: {condition_filter}")
        
        try:
            if self.use_azure_search and self.azure_search_client:
                return self._azure_search(search_query, condition_filter, max_results)
            else:
                return self._vector_search(search_query, condition_filter, max_results)
                
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def _vector_search(
        self,
        query: str,
        condition_filter: str,
        max_results: int
    ) -> List[SearchResult]:
        """Perform vector search using existing FAISS retriever"""
        
        if not self.vector_retriever:
            self.logger.warning("No vector retriever available")
            return []
        
        # Get relevant documents
        docs = self.vector_retriever.get_relevant_documents(query)
        
        # Apply condition filter
        if condition_filter != "ALL":
            docs = [
                doc for doc in docs 
                if condition_filter.upper() in doc.page_content.upper()
            ]
        
        # Convert to SearchResult objects
        results = []
        for i, doc in enumerate(docs[:max_results]):
            result = SearchResult(
                content=doc.page_content,
                metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                score=1.0 - (i * 0.1),  # Simple scoring based on rank
                source="vector_search"
            )
            results.append(result)
        
        self.logger.info(f"Vector search returned {len(results)} results")
        return results
    
    def _azure_search(
        self,
        query: str,
        condition_filter: str,
        max_results: int
    ) -> List[SearchResult]:
        """Perform Azure AI Search (for future implementation)"""
        
        # Placeholder for Azure Search implementation
        self.logger.info("Azure Search not yet implemented, falling back to vector search")
        return self._vector_search(query, condition_filter, max_results)
    
    def get_related_queries(self, original_query: str, results: List[SearchResult]) -> List[str]:
        """Generate related queries based on search results"""
        
        if not results:
            return []
        
        # Extract key terms from top results
        top_content = " ".join([r.content[:200] for r in results[:3]])
        
        # Generate related queries based on clinical context
        related_queries = []
        
        # Clinical condition-based queries
        if any(term in original_query.lower() for term in ["pressure", "injury", "ulcer"]):
            related_queries.extend([
                "Braden scale assessment pressure injury",
                "Hospital acquired pressure injury prevention",
                "Risk factors for pressure ulcers"
            ])
        
        if any(term in original_query.lower() for term in ["kidney", "renal", "aki"]):
            related_queries.extend([
                "Acute kidney injury KDIGO criteria",
                "Hospital acquired acute kidney injury prevention",
                "Nephrotoxic medications AKI risk"
            ])
        
        if any(term in original_query.lower() for term in ["anemia", "hemoglobin", "hematocrit"]):
            related_queries.extend([
                "Hospital acquired anemia causes",
                "Anemia management in hospitalized patients",
                "Blood loss anemia hospital setting"
            ])
        
        return related_queries[:3]  # Return top 3 related queries

class ClinicalSearchOrchestrator:
    """Orchestrates search across multiple sources and enhances results"""
    
    def __init__(self, search_function: SearchVectorFunction, llm_client: OpenAI):
        self.search_function = search_function
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def enhanced_search(
        self,
        query: str,
        include_related: bool = True,
        condition_filter: str = "ALL"
    ) -> Dict[str, Any]:
        """
        Perform enhanced search with query expansion and result synthesis
        """
        
        # Primary search
        primary_results = self.search_function.search(
            query, condition_filter=condition_filter, max_results=6
        )
        
        enhanced_response = {
            "primary_results": [self._serialize_result(r) for r in primary_results],
            "query": query,
            "condition_filter": condition_filter,
            "total_results": len(primary_results)
        }
        
        # Add related queries if requested
        if include_related and primary_results:
            related_queries = self.search_function.get_related_queries(query, primary_results)
            enhanced_response["related_queries"] = related_queries
            
            # Perform searches for related queries
            related_results = []
            for related_query in related_queries[:2]:  # Limit to 2 related searches
                related_res = self.search_function.search(
                    related_query, condition_filter=condition_filter, max_results=3
                )
                if related_res:
                    related_results.extend(related_res[:2])  # Take top 2 from each
            
            enhanced_response["related_results"] = [
                self._serialize_result(r) for r in related_results
            ]
        
        # Generate search summary
        if primary_results:
            enhanced_response["search_summary"] = self._generate_search_summary(
                query, primary_results
            )
        
        return enhanced_response
    
    def _serialize_result(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to serializable dictionary"""
        return {
            "content": result.content,
            "metadata": result.metadata,
            "score": result.score,
            "source": result.source
        }
    
    def _generate_search_summary(self, query: str, results: List[SearchResult]) -> str:
        """Generate a brief summary of search results"""
        
        if len(results) == 0:
            return "No relevant results found."
        
        # Create a concise summary
        summary_parts = []
        summary_parts.append(f"Found {len(results)} relevant clinical records")
        
        # Identify common themes
        all_content = " ".join([r.content for r in results[:3]])
        if "pressure injury" in all_content.lower():
            summary_parts.append("related to hospital-acquired pressure injuries")
        elif "kidney injury" in all_content.lower():
            summary_parts.append("related to acute kidney injury")
        elif "anemia" in all_content.lower():
            summary_parts.append("related to hospital-acquired anemia")
        
        return " ".join(summary_parts) + "."

# Function specification for Smart Agent integration
SEARCH_FUNCTION_SPEC = {
    "type": "function",
    "function": {
        "name": "search_clinical_records",
        "description": "Search clinical records using semantic vector search with condition filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Natural language query about clinical conditions or patient care"
                },
                "condition_filter": {
                    "type": "string",
                    "enum": ["HAPI", "HAAKI", "HAA", "ALL"],
                    "description": "Filter results by specific hospital-acquired condition",
                    "default": "ALL"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 6,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["search_query"]
        }
    }
}