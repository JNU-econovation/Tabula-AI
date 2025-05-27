from typing import List, Optional
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from .models import SearchResult, HybridSearchResponse, RetrievalConfig

from common_sdk import get_embedding, get_logger
from ..config import settings

logger = get_logger()

class PineconeSearcher:
    """Pinecone을 사용한 벡터 검색 클래스"""
    
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = settings.INDEX_NAME
        self.logger = get_logger()
        self._index = None  # 인덱스 캐싱용

    def get_index(self, index_name: Optional[str] = None):
        """Pinecone 인덱스 객체 반환"""
        target_index_name = index_name or self.index_name
        
        if self._index is None or getattr(self._index, 'name', None) != target_index_name:
            self._index = self.pc.Index(target_index_name)
        
        return self._index
    
    def list_indexes(self):
        """사용 가능한 인덱스 목록 반환"""
        try:
            indexes = self.pc.list_indexes()
            return [index.name for index in indexes]
        except Exception as e:
            self.logger.error(f"Failed to list indexes: {str(e)}")
            return []
    
    def index_exists(self, index_name: str) -> bool:
        """인덱스 존재 여부 확인"""
        try:
            available_indexes = self.list_indexes()
            return index_name in available_indexes
        except Exception as e:
            self.logger.error(f"Failed to check index existence: {str(e)}")
            return False
    
    def get_index_info(self, index_name: str):
        """인덱스 정보 반환"""
        try:
            if not self.index_exists(index_name):
                return None
            
            index = self.get_index(index_name)
            stats = index.describe_index_stats()
            return {
                "name": index_name,
                "dimension": stats.dimension if hasattr(stats, 'dimension') else 'unknown',
                "total_vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0,
                "namespaces": stats.namespaces if hasattr(stats, 'namespaces') else {}
            }
        except Exception as e:
            self.logger.error(f"Failed to get index info: {str(e)}")
            return None
    
    def hybrid_search(self, 
        query: str, 
        config: RetrievalConfig,
        bm25_encoder: BM25Encoder,
        language: str = "ko"
    ) -> HybridSearchResponse:
        """하이브리드 검색 (Dense + Sparse) 수행"""
        try:
            # Dense 벡터 생성
            dense_vector = get_embedding(query, language=language)
            
            # Sparse 벡터 생성
            sparse_vector = bm25_encoder.encode_queries([query])[0]
            
            # Pinecone 인덱스 객체 가져오기
            index = self.get_index(config.index_name)
            filter_dict = {"document_id": config.document_id}
            
            results = index.query(
                vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=config.top_k,
                include_metadata=True,
                filter=filter_dict,
                alpha=config.alpha
            )
            
            # 결과 변환
            search_results = []
            if results and results.matches:
                for match in results.matches:
                    search_results.append(SearchResult(
                        content=match.metadata.get("content", ""),
                        score=match.score,
                        metadata=match.metadata
                    ))
            
            return HybridSearchResponse(
                matches=search_results,
                total_count=len(search_results)
            )
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {str(e)}")
            return HybridSearchResponse(matches=[], total_count=0)
    
class BM25Manager:
    """BM25 인코더 관리 클래스"""
    
    def __init__(self):
        self.encoder = BM25Encoder()
        self.logger = get_logger()
    
    def fit(self, texts: List[str]) -> None:
        """텍스트 리스트로 BM25 인코더 훈련"""
        try:
            self.encoder.fit(texts)
            self.logger.info(f"BM25 encoder fitted with {len(texts)} texts")
        except Exception as e:
            self.logger.error(f"BM25 encoder fit error: {str(e)}")
            raise
    
    def encode_queries(self, queries: List[str]) -> List:
        """쿼리 리스트를 sparse 벡터로 인코딩"""
        try:
            return self.encoder.encode_queries(queries)
        except Exception as e:
            self.logger.error(f"BM25 encode queries error: {str(e)}")
            raise
    
    def get_encoder(self) -> BM25Encoder:
        """BM25 인코더 반환"""
        return self.encoder
    
class DocumentFinder:
    """문서 검색을 위한 통합 클래스"""
    
    def __init__(self):
        self.pinecone_searcher = PineconeSearcher()
        self.bm25_manager = BM25Manager()
        self.logger = get_logger()
    
    def setup_bm25(self, texts: List[str]) -> None:
        """BM25 인코더 초기화"""
        self.bm25_manager.fit(texts)
    
    def find_reference_text(
        self, 
        query: str, 
        config: RetrievalConfig,
        language: str = "ko"
    ) -> Optional[str]:
        """참고 텍스트 검색"""
        try:
            response = self.pinecone_searcher.hybrid_search(
                query=query,
                config=config,
                bm25_encoder=self.bm25_manager.get_encoder(),
                language=language
            )
            
            if response.matches and len(response.matches) > 0:
                return response.matches[0].content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Reference text search error: {str(e)}")
            return None
    
    def get_bm25_encoder(self) -> BM25Encoder:
        """BM25 인코더 반환"""
        return self.bm25_manager.get_encoder()