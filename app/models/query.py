from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str
    filters: Dict[str, Any] = {}
    top_k: int = 5

class EvaluationMetrics(BaseModel):
    """Evaluation metrics for RAG retrieval and generation quality"""
    avg_retrieval_score: float  # Average similarity score of retrieved documents
    max_retrieval_score: float  # Highest similarity score
    min_retrieval_score: float  # Lowest similarity score
    num_sources_used: int       # Number of documents retrieved
    confidence_score: float     # Overall confidence in the answer (0-1)
    coverage_score: float       # How well the sources cover the question (0-1)
    source_diversity: float     # Diversity of source documents (0-1)

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    reasoning: str
    evaluation_metrics: Optional[EvaluationMetrics] = None
