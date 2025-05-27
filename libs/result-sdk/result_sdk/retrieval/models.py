from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchResult(BaseModel):
    """검색 결과를 나타내는 모델"""
    content: str
    score: float
    metadata: Dict[str, Any]


class HybridSearchResponse(BaseModel):
    """하이브리드 검색 응답 모델"""
    matches: List[SearchResult]
    total_count: int


class RetrievalConfig(BaseModel):
    """검색 설정 모델"""
    top_k: int = 3
    alpha: float = 0.7  # Dense와 Sparse 벡터 가중치
    document_id: str
    index_name: str