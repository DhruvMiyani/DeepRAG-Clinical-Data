import logging
import sys
import pandas as pd
from typing import Optional, Dict, Any, List

from config import Config
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(stream=sys.stdout, level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline for clinical data processing."""
    
    def __init__(self):
        """Initialize the RAG pipeline with configuration."""
        if not Config.validate_config():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        self.llm = self._initialize_llm()
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.parser = StrOutputParser()
        
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        return ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            streaming=True
        )
    
    def create_pandas_query_engine(self, df: pd.DataFrame) -> PandasQueryEngine:
        """Create a query engine for pandas DataFrame."""
        llm = OpenAI(
            model=Config.DEFAULT_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        return PandasQueryEngine(df=df, llm=llm, verbose=True)
    
    def create_pandas_agent(self, df: pd.DataFrame):
        """Create a pandas DataFrame agent for complex queries."""
        return create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True
        )
    
    def create_vector_store(self, text_data: List[str]) -> DocArrayInMemorySearch:
        """Create a vector store from text data."""
        return DocArrayInMemorySearch.from_texts(
            text_data,
            embedding=self.embeddings
        )
    
    def setup_retrieval_chain(self, vector_store: DocArrayInMemorySearch):
        """Set up the retrieval chain with the vector store."""
        retriever = vector_store.as_retriever()
        setup = RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        return setup
    
    def calculate_similarity(self, query: str, sentences: List[str]) -> List[float]:
        """Calculate cosine similarity between query and sentences."""
        embedded_query = self.embeddings.embed_query(query)
        embedded_sentences = [self.embeddings.embed_query(s) for s in sentences]
        
        similarities = [
            cosine_similarity([embedded_query], [sentence])[0][0]
            for sentence in embedded_sentences
        ]
        return similarities
    
    def process_dataframe(self, df: pd.DataFrame, id_col: str, time_col: str) -> List[str]:
        """Process DataFrame for vector store creation."""
        df['combined'] = df[id_col].astype(str) + " at " + df[time_col].astype(str)
        return df['combined'].tolist()


class DataLoader:
    """Utility class for loading various data formats."""
    
    SUPPORTED_FORMATS = {
        "csv": pd.read_csv,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "xlsm": pd.read_excel,
        "xlsb": pd.read_excel,
    }
    
    @classmethod
    def load_data(cls, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from file based on extension."""
        try:
            ext = file_path.split(".")[-1].lower()
            if ext in cls.SUPPORTED_FORMATS:
                return cls.SUPPORTED_FORMATS[ext](file_path)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None


def main():
    """Main execution function."""
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully")
        
        # Example usage (replace with actual data loading)
        # df = DataLoader.load_data("path_to_your_data.csv")
        # if df is not None:
        #     text_data = pipeline.process_dataframe(df, 'hadm_id', 'admittime')
        #     vector_store = pipeline.create_vector_store(text_data)
        #     retrieval_chain = pipeline.setup_retrieval_chain(vector_store)
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()