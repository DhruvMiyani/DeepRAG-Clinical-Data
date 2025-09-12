"""Custom exceptions for the RAG Clinical Data application."""


class RAGException(Exception):
    """Base exception for RAG application."""
    pass


class ConfigurationError(RAGException):
    """Raised when there's a configuration error."""
    pass


class DataLoadingError(RAGException):
    """Raised when data loading fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    pass


class ModelError(RAGException):
    """Raised when LLM model operations fail."""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class APIError(RAGException):
    """Raised when external API calls fail."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class EvaluationError(RAGException):
    """Raised when evaluation operations fail."""
    pass