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