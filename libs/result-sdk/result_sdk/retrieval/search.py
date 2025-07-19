# retrieval/search.py
import json
import asyncio
from typing import List, Optional, Dict, Any
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from .models import SearchResult, HybridSearchResponse, RetrievalConfig

from common_sdk import get_embedding, get_logger
from ..config import settings

logger = get_logger()

class PineconeSearcher:
    """Pinecone을 사용한 벡터 검색 클래스 (비동기 지원)"""
    
    def __init__(self, language: str = "korean"):
        self.api_key = settings.PINECONE_API_KEY
        self.pc = Pinecone(api_key=self.api_key)
        self.language = language
        self.logger = get_logger()
        
        # 언어별 Dense 차원 설정
        if language == "korean":
            self.dense_vector_dimension = 4096  # Upstage 임베딩
        else:  # "english"
            self.dense_vector_dimension = 3072  # OpenAI 3-large 임베딩
        
        # Dense/Sparse 인덱스 이름 설정
        if language == "korean":
            self.dense_index_name = settings.INDEX_NAME_KOR_DEN_CONTENTS
            self.sparse_index_name = settings.INDEX_NAME_KOR_SPA_CONTENTS
        else:
            self.dense_index_name = settings.INDEX_NAME_ENG_DEN_CONTENTS
            self.sparse_index_name = settings.INDEX_NAME_ENG_SPA_CONTENTS
        
        self._dense_index = None  # Dense 인덱스 캐싱용
        self._sparse_index = None  # Sparse 인덱스 캐싱용

    def get_dense_index(self):
        """Dense Pinecone 인덱스 객체 반환"""
        if self._dense_index is None:
            self._dense_index = self.pc.Index(self.dense_index_name)
        return self._dense_index
    
    def get_sparse_index(self):
        """Sparse Pinecone 인덱스 객체 반환"""
        if self._sparse_index is None:
            self._sparse_index = self.pc.Index(self.sparse_index_name)
        return self._sparse_index
    
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
            
            index = self.pc.Index(index_name)
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
    
    async def dense_search(self, 
        query: str, 
        config: RetrievalConfig,
        language: str = "korean"
    ) -> List[SearchResult]:
        """Dense 벡터 검색 수행 (비동기)"""
        try:
            # Dense 벡터 생성
            dense_vector = await asyncio.to_thread(get_embedding, query, language=language)
            
            # Dense 인덱스에서 검색
            dense_index = self.get_dense_index()
            filter_dict = {"spaceId": config.space_id}
            
            results = await asyncio.to_thread(
                dense_index.query,
                vector=dense_vector,
                top_k=config.top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace="documents"
            )
            
            # 결과 변환 및 이미지 콘텐츠 추가
            search_results = []
            if results and results.matches:
                for match in results.matches:
                    enhanced_content = await self.enhance_content_with_images(
                        match.metadata.get("content", ""),
                        match.metadata,
                        config.space_id
                    )
                    search_results.append(SearchResult(
                        content=enhanced_content,
                        score=match.score,
                        metadata=match.metadata
                    ))
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Dense search error: {str(e)}")
            return []
    
    async def sparse_search(self, 
        query: str, 
        config: RetrievalConfig,
        bm25_encoder: BM25Encoder
    ) -> List[SearchResult]:
        """Sparse 벡터 검색 수행 (비동기)"""
        try:
            # Sparse 벡터 생성
            sparse_vector = await asyncio.to_thread(
                bm25_encoder.encode_queries, 
                [query]
            )
            sparse_vector = sparse_vector[0]
            
            # Sparse 인덱스에서 검색
            sparse_index = self.get_sparse_index()
            filter_dict = {"spaceId": config.space_id}
            
            # sparse_values 형태로 변환
            sparse_values = {
                "indices": sparse_vector['indices'],
                "values": sparse_vector['values']
            }
            
            results = await asyncio.to_thread(
                sparse_index.query,
                sparse_vector=sparse_values,
                top_k=config.top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace="documents"
            )
            
            # 결과 변환 및 이미지 콘텐츠 추가
            search_results = []
            if results and results.matches:
                for match in results.matches:
                    enhanced_content = await self.enhance_content_with_images(
                        match.metadata.get("content", ""),
                        match.metadata,
                        config.space_id
                    )
                    search_results.append(SearchResult(
                        content=enhanced_content,
                        score=match.score,
                        metadata=match.metadata
                    ))
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Sparse search error: {str(e)}")
            return []
    
    async def hybrid_search(self, 
        query: str, 
        config: RetrievalConfig,
        bm25_encoder: BM25Encoder,
        language: str = "korean"
    ) -> HybridSearchResponse:
        """하이브리드 검색 (Dense + Sparse) 수행 (비동기)"""
        try:
            # Dense와 Sparse 검색 병렬로 수행
            dense_results, sparse_results = await asyncio.gather(
                self.dense_search(query, config, language),
                self.sparse_search(query, config, bm25_encoder)
            )
            
            # 결과 합치기 및 가중치 적용
            combined_results = self.combine_results(
                dense_results, 
                sparse_results, 
                config.alpha
            )
            
            # top_k만큼 반환
            final_results = combined_results[:config.top_k]
            
            return HybridSearchResponse(
                matches=final_results,
                total_count=len(final_results)
            )
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {str(e)}")
            return HybridSearchResponse(matches=[], total_count=0)
    
    def combine_results(self, 
        dense_results: List[SearchResult], 
        sparse_results: List[SearchResult], 
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """Dense와 Sparse 결과를 조합"""
        try:
            # ID별로 결과 정리
            combined_scores = {}
            
            # Dense 결과 처리
            for result in dense_results:
                result_id = self.get_result_id(result.metadata)
                combined_scores[result_id] = {
                    'content': result.content,
                    'metadata': result.metadata,
                    'dense_score': result.score * alpha,
                    'sparse_score': 0.0
                }
            
            # Sparse 결과 처리
            for result in sparse_results:
                result_id = self.get_result_id(result.metadata)
                if result_id in combined_scores:
                    combined_scores[result_id]['sparse_score'] = result.score * (1 - alpha)
                else:
                    combined_scores[result_id] = {
                        'content': result.content,
                        'metadata': result.metadata,
                        'dense_score': 0.0,
                        'sparse_score': result.score * (1 - alpha)
                    }
            
            # 최종 점수 계산 및 정렬
            final_results = []
            for result_data in combined_scores.values():
                final_score = result_data['dense_score'] + result_data['sparse_score']
                final_results.append(SearchResult(
                    content=result_data['content'],
                    score=final_score,
                    metadata=result_data['metadata']
                ))
            
            # 점수순으로 정렬
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Results combination error: {str(e)}")
            return []
    
    def get_result_id(self, metadata: Dict[str, Any]) -> str:
        """결과의 고유 ID 생성"""
        space_id = metadata.get('spaceId', '')
        chunk_id = metadata.get('chunkId', '')
        content_type = metadata.get('type', '')
        return f"{space_id}_{chunk_id}_{content_type}"
    
    async def enhance_content_with_images(self, text_content: str, metadata: Dict[str, Any], space_id: str) -> str:
        """텍스트 콘텐츠에 관련 이미지 설명 추가 (비동기)"""
        try:
            # imageReferences가 있는지 확인
            image_references_str = metadata.get('imageReferences', '')
            
            if not image_references_str:
                return text_content
            
            # JSON 문자열 파싱하여 리스트로 변환
            try:
                image_references = json.loads(image_references_str)
            except (json.JSONDecodeError, TypeError):
                self.logger.error(f"Failed to parse imageReferences JSON: {image_references_str}")
                return text_content
            
            # 이미지 콘텐츠 조회
            image_tasks = []
            for image_ref in image_references:

                if isinstance(image_ref, dict) and 'imagePath' in image_ref:
                    image_path = image_ref['imagePath']
                    task = self.get_image_content(image_path, space_id)
                    image_tasks.append(task)
                elif isinstance(image_ref, str):
                    # 문자열로 저장된 경우도 처리
                    task = self.get_image_content(image_ref, space_id)
                    image_tasks.append(task)
            
            # 모든 이미지 콘텐츠를 병렬로 조회
            if image_tasks:
                image_contents_raw = await asyncio.gather(*image_tasks)
                image_contents = [f"[이미지 설명: {content}]" for content in image_contents_raw if content]
            else:
                image_contents = []
            
            # 텍스트 콘텐츠와 이미지 설명 결합
            if image_contents:
                enhanced_content = text_content + "\n\n" + "\n".join(image_contents)
                return enhanced_content
            
            return text_content
            
        except Exception as e:
            self.logger.error(f"Error enhancing content with images: {str(e)}")
            return text_content
    
    async def get_image_content(self, image_path: str, space_id: str) -> Optional[str]:
        """특정 이미지 경로에 해당하는 이미지 콘텐츠 조회 (비동기)"""
        try:
            # Dense 인덱스에서 이미지 타입 검색
            dense_index = self.get_dense_index()
            
            # metadata가 이미지 타입 + 해당 imagePath를 가진 벡터 검색
            filter_dict = {
                "spaceId": space_id,
                "type": "image",
                "imagePath": image_path
            }
            
            # Dense 차원에 맞는 더미 벡터 생성
            dummy_vector = [0.0] * self.dense_vector_dimension
            
            results = await asyncio.to_thread(
                dense_index.query,
                vector=dummy_vector,  # 언어별 Dense 차원 사용
                top_k=1,
                include_metadata=True,
                filter=filter_dict,
                namespace="documents"
            )
            
            if results and results.matches and len(results.matches) > 0:
                return results.matches[0].metadata.get("content", "")
            
            # 이미지를 찾지 못한 경우 경고만 로그
            self.logger.warning(f"No image found for path: {image_path} in space: {space_id}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Image content not found for {image_path}: {str(e)}")
            return None

class BM25Manager:
    """BM25 인코더 관리 클래스"""
    
    def __init__(self):
        self.encoder = BM25Encoder()
        self.logger = get_logger()
    
    async def fit(self, texts: List[str]) -> None:
        """텍스트 리스트로 BM25 인코더 훈련 (비동기)"""
        try:
            await asyncio.to_thread(self.encoder.fit, texts)
            self.logger.info(f"BM25 encoder fitted with {len(texts)} texts")
        except Exception as e:
            self.logger.error(f"BM25 encoder fit error: {str(e)}")
            raise
    
    async def encode_queries(self, queries: List[str]) -> List:
        """쿼리 리스트를 sparse 벡터로 인코딩 (비동기)"""
        try:
            return await asyncio.to_thread(self.encoder.encode_queries, queries)
        except Exception as e:
            self.logger.error(f"BM25 encode queries error: {str(e)}")
            raise
    
    def get_encoder(self) -> BM25Encoder:
        """BM25 인코더 반환"""
        return self.encoder
    
class DocumentFinder:
    """문서 검색을 위한 통합 클래스"""
    
    def __init__(self, language: str = "korean"):
        self.pinecone_searcher = PineconeSearcher(language)
        self.bm25_manager = BM25Manager()
        self.logger = get_logger()
    
    async def setup_bm25(self, texts: List[str]) -> None:
        """BM25 인코더 초기화 (비동기)"""
        await self.bm25_manager.fit(texts)
    
    async def find_reference_text(
        self, 
        query: str, 
        config: RetrievalConfig,
        language: str = "korean"
    ) -> Optional[str]:
        """참고 텍스트 검색 (비동기 / Dense)"""
        try:
            dense_results = await self.pinecone_searcher.dense_search(
                query=query,
                config=config,
                language=language
            )
            
            if dense_results and len(dense_results) > 0:
                return dense_results[0].content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Reference text search error: {str(e)}")
            return None
    
    def get_bm25_encoder(self) -> BM25Encoder:
        """BM25 인코더 반환"""
        return self.bm25_manager.get_encoder()