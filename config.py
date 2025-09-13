import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration management for the RAG Clinical Data application."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))
    
    # Available GPT models (in order of capability)
    AVAILABLE_MODELS = [
        "gpt-5",                # Latest GPT-5 model
        "gpt-4-turbo-preview",  # Latest GPT-4 Turbo with 128K context
        "gpt-4-1106-preview",   # GPT-4 Turbo
        "gpt-4",                # Standard GPT-4
        "gpt-4-32k",            # GPT-4 with 32K context
        "gpt-3.5-turbo-1106",   # Latest GPT-3.5 Turbo
        "gpt-3.5-turbo",        # Standard GPT-3.5 Turbo
    ]
    
    # Application Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "750"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "15"))
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Retriever Configuration
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))
    PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "750"))
    CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", "200"))
    
    # Azure AI Search Configuration (for DeepRAG)
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "clinical-index")
    
    # Azure Cosmos DB Configuration (for Graph RAG)
    COSMOS_DB_ENDPOINT: str = os.getenv("COSMOS_DB_ENDPOINT", "")
    COSMOS_DB_KEY: str = os.getenv("COSMOS_DB_KEY", "")
    COSMOS_DB_DATABASE: str = os.getenv("COSMOS_DB_DATABASE", "clinical-graph")
    COSMOS_DB_CONTAINER: str = os.getenv("COSMOS_DB_CONTAINER", "clinical-entities")
    
    # Azure Storage Configuration
    AZURE_STORAGE_ACCOUNT_NAME: str = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    AZURE_STORAGE_ACCOUNT_KEY: str = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
    AZURE_CONTAINER_NAME: str = os.getenv("AZURE_CONTAINER_NAME", "clinical-docs")
    
    # Redis Cache Configuration
    AZURE_REDIS_ENDPOINT: str = os.getenv("AZURE_REDIS_ENDPOINT", "")
    AZURE_REDIS_KEY: str = os.getenv("AZURE_REDIS_KEY", "")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6380"))
    
    # DeepRAG Agent Configuration
    SMART_AGENT_PROMPT_LOCATION: str = os.getenv("SMART_AGENT_PROMPT_LOCATION", "./prompts/smart_agent.yaml")
    MAX_RUN_PER_QUESTION: int = int(os.getenv("MAX_RUN_PER_QUESTION", "10"))
    MAX_QUESTION_TO_KEEP: int = int(os.getenv("MAX_QUESTION_TO_KEEP", "3"))
    MAX_QUESTION_WITH_DETAIL_HIST: int = int(os.getenv("MAX_QUESTION_WITH_DETAIL_HIST", "1"))
    
    # Feature Flags
    ENABLE_DEEPRAG: bool = os.getenv("ENABLE_DEEPRAG", "false").lower() == "true"
    ENABLE_AZURE_SEARCH: bool = os.getenv("ENABLE_AZURE_SEARCH", "false").lower() == "true"
    ENABLE_GRAPH_RAG: bool = os.getenv("ENABLE_GRAPH_RAG", "false").lower() == "true"
    ENABLE_REDIS_CACHE: bool = os.getenv("ENABLE_REDIS_CACHE", "false").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration values are set."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        return True
    
    @classmethod
    def get_openai_client(cls):
        """Get configured OpenAI client."""
        from openai import OpenAI
        
        if not cls.validate_config():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        return OpenAI(api_key=cls.OPENAI_API_KEY)
    
    @classmethod
    def get_langchain_llm(cls):
        """Get configured LangChain LLM."""
        from langchain_openai import ChatOpenAI
        
        if not cls.validate_config():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        return ChatOpenAI(
            openai_api_key=cls.OPENAI_API_KEY,
            model=cls.DEFAULT_MODEL,
            temperature=cls.DEFAULT_TEMPERATURE
        )
    
    @classmethod
    def get_azure_search_client(cls):
        """Get configured Azure AI Search client."""
        if not cls.AZURE_SEARCH_ENDPOINT or not cls.AZURE_SEARCH_KEY:
            return None
        
        try:
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential
            
            return SearchClient(
                endpoint=cls.AZURE_SEARCH_ENDPOINT,
                index_name=cls.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(cls.AZURE_SEARCH_KEY)
            )
        except ImportError:
            print("Azure Search SDK not installed. Install with: pip install azure-search-documents")
            return None
    
    @classmethod
    def get_cosmos_client(cls):
        """Get configured Azure Cosmos DB client."""
        if not cls.COSMOS_DB_ENDPOINT or not cls.COSMOS_DB_KEY:
            return None
        
        try:
            from azure.cosmos import CosmosClient
            
            cosmos_client = CosmosClient(
                cls.COSMOS_DB_ENDPOINT,
                cls.COSMOS_DB_KEY
            )
            database = cosmos_client.get_database_client(cls.COSMOS_DB_DATABASE)
            return database.get_container_client(cls.COSMOS_DB_CONTAINER)
        except ImportError:
            print("Azure Cosmos SDK not installed. Install with: pip install azure-cosmos")
            return None
    
    @classmethod
    def get_redis_client(cls):
        """Get configured Redis client."""
        if not cls.AZURE_REDIS_ENDPOINT or not cls.AZURE_REDIS_KEY:
            return None
        
        try:
            import redis
            
            return redis.Redis(
                host=cls.AZURE_REDIS_ENDPOINT,
                port=cls.REDIS_PORT,
                password=cls.AZURE_REDIS_KEY,
                ssl=True,
                decode_responses=True
            )
        except ImportError:
            print("Redis SDK not installed. Install with: pip install redis")
            return None