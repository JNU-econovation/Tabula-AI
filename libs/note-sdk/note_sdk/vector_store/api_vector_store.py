import re
import asyncio

from pinecone import Pinecone
from typing import List, Dict, Any, Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from pinecone_text.sparse import BM25Encoder

from common_sdk.utils import get_embedding
from note_sdk.config import settings
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()


class VectorLoader:
    def __init__(self, language: Literal["ko", "en"], space_id: str = None):
        """
        Args:
            language: 언어 설정 (ko / en)
            space_id: 작업 ID
        
        VectorLoader 초기화
        """
        self.language = language
        self.space_id = space_id
        
        # Pinecone 클라이언트 초기화(language에 따른 API KEY 설정)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # 인덱스 초기화
        self.dense_index = self.init_dense_index()
        self.sparse_index = self.init_sparse_index()
        
        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 스레드 풀 초기화
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 처리된 벡터 수 추적 변수
        self.processed_vectors = {
            "text": {"dense": 0, "sparse": 0},
            "image": {"dense": 0, "sparse": 0}
        }

    # Dense Vector Pinecone 인덱스 초기화
    def init_dense_index(self) -> Any:
        try:
            # 언어에 따른 인덱스 설정
            index_name = settings.INDEX_NAME_KOR_DEN_CONTENTS if self.language == "ko" else settings.INDEX_NAME_ENG_DEN_CONTENTS
            
            # 인덱스 존재 확인
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"'{index_name}' index not found")
                # 인덱스가 없으면 생성
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,  # dense vector 차원
                    metric="cosine"  # dense vector는 cosine similarity 사용
                )
                logger.info(f"Created new dense index: {index_name}")
            
            return self.pc.Index(index_name)
        
        except Exception as e:
            logger.error(f"Dense index initialization error: {e}")
            return None

    # Sparse Vector Pinecone 인덱스 초기화
    def init_sparse_index(self) -> Any:
        try:
            # 언어에 따른 인덱스 설정
            index_name = settings.INDEX_NAME_KOR_SPA_CONTENTS if self.language == "ko" else settings.INDEX_NAME_ENG_SPA_CONTENTS
            
            # 인덱스 존재 확인
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"'{index_name}' index not found")
                # 인덱스가 없으면 생성
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,  # sparse vector 차원
                    metric="dotproduct"
                )
                logger.info(f"Created new sparse index: {index_name}")
            
            return self.pc.Index(index_name)
        
        except Exception as e:
            logger.error(f"Sparse index initialization error: {e}")
            return None
        
    # 마크다운에서 이미지 경로 추출
    def extract_image_paths(self, content: str) -> List[Dict[str, str]]:
        pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.finditer(pattern, content)
        # alt_text 제거하고 image_path만 반환
        return [{"image_path": m.group(2)} for m in matches]

    # 청크에 포함된 이미지 경로 찾기
    def find_images_in_chunk(self, chunk: str, image_paths: List[Dict[str, str]]) -> List[Dict[str, str]]:

        return [img for img in image_paths if img["image_path"] in chunk]
    

    # 청크 처리 및 벡터 DB 적재
    async def process_chunk(self, chunk: str, chunk_id: int, document_id: str, image_paths: List[Dict[str, str]]) -> bool:
        try:
            # 1. 청크에 포함된 이미지 경로 찾기
            chunk_image_paths = self.find_images_in_chunk(chunk, image_paths)
            
            # 2. 임베딩 생성 (Dense/Sparse)
            dense_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                get_embedding,
                chunk,
                self.language
            )
            sparse_embedding = self.bm25.encode_documents([chunk])[0]  # {'indices': [...], 'values': [...]} 형식
            
            # 3. 메타데이터 구성
            metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "type": "text",
                "content": chunk,
                "image_references": [img["image_path"] for img in chunk_image_paths]
            }
            
            # 4. 벡터 데이터 구성
            dense_vector = {
                "id": f"{document_id}_{self.language}_{chunk_id}_text_dense",
                "values": dense_embedding,
                "metadata": metadata
            }
            sparse_vector = {
                "id": f"{document_id}_{self.language}_{chunk_id}_text_sparse",
                "sparse_values": {
                    "indices": sparse_embedding['indices'],
                    "values": sparse_embedding['values']
                },
                "metadata": metadata
            }
            
            # 5. Dense/Sparse 벡터 병렬 적재
            dense_task = asyncio.create_task(self.upsert_dense_vector(dense_vector, document_id))
            sparse_task = asyncio.create_task(self.upsert_sparse_vector(document_id, sparse_vector, metadata))
            
            await dense_task
            await sparse_task
            
            return True
            
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            return False

    async def upsert_dense_vector(self, vector: Dict[str, Any], namespace: str) -> bool:
        """Dense 벡터 적재"""
        try:
            # namespace를 'documents'로 통일하고 document_id는 메타데이터로 유지
            self.dense_index.upsert(vectors=[vector], namespace="documents")
            self.processed_vectors["text"]["dense"] += 1
            return True
        except Exception as e:
            logger.error(f"Dense vector upsertion error: {e}")
            return False

    def convert_sparse_vector_format(self, sparse_vector: Dict) -> List:
        """sparse vector 형식을 Pinecone이 기대하는 형식으로 변환"""
        try:
            if not isinstance(sparse_vector, dict):
                logger.error(f"Invalid sparse vector type: {type(sparse_vector)}")
                return []
            
            indices = sparse_vector.get('indices', [])
            values = sparse_vector.get('values', [])
            
            # 디버깅을 위한 로그 추가
            logger.info(f"Sparse vector indices: {indices[:5]}...")
            logger.info(f"Sparse vector values: {values[:5]}...")
            
            # indices와 values를 리스트로 변환
            if isinstance(indices, list) and isinstance(values, list):
                if len(indices) != len(values):
                    logger.error(f"Length mismatch: indices={len(indices)}, values={len(values)}")
                    return []
                return list(zip(indices, values))
            
            logger.error(f"Invalid indices or values type: indices={type(indices)}, values={type(values)}")
            return []
            
        except Exception as e:
            logger.error(f"Error in sparse vector format conversion: {str(e)}")
            return []

    async def upsert_sparse_vector(self, document_id: str, sparse_vector: Dict, metadata: Dict = None) -> bool:
        """sparse vector를 Pinecone에 업서트"""
        try:
            # Pinecone에 업서트 (sparse_values 사용)
            self.sparse_index.upsert(
                vectors=[{
                    'id': sparse_vector['id'],
                    'sparse_values': sparse_vector['sparse_values'],
                    'metadata': metadata or {}
                }],
                namespace='documents'
            )
            self.processed_vectors["text"]["sparse"] += 1
            return True
        except Exception as e:
            logger.error(f"Sparse vector upsertion error: {str(e)}")
            return False

    # 마크다운 처리 및 벡터 DB 적재
    async def process_markdown(self, content: str, document_id: str) -> bool:
        try:
            # 1. 이미지 경로 추출
            image_paths = self.extract_image_paths(content)
            
            # 2. 텍스트 분할
            chunks = self.text_splitter.split_text(content)
            
            # 3. BM25 인코더 초기화
            self.bm25 = BM25Encoder()
            self.bm25.fit(chunks)
            
            # 4. 청크별 파이프라인 처리
            tasks = []
            for i, chunk in enumerate(chunks):
                task = asyncio.create_task(
                    self.process_chunk(chunk, i + 1, document_id, image_paths)
                )
                tasks.append(task)
            
            # 5. 모든 청크 처리 완료 대기
            results = await asyncio.gather(*tasks)
            return all(results)
            
        except Exception as e:
            logger.error(f"Markdown processing error: {e}")
            return False

    # 이미지 처리도 비슷한 방식으로 수정
    async def process_image(self, image_info: Dict[str, Any], document_id: str) -> bool:
        try:
            # 1. 임베딩 생성
            dense_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                get_embedding,
                image_info["summary"],
                self.language
            )
            sparse_embedding = self.bm25.encode_documents([image_info["summary"]])[0]
            
            # 2. 메타데이터 구성
            metadata = {
                "document_id": document_id,
                "chunk_id": image_info["chunk_id"],
                "type": "image",
                "content": image_info["summary"],
                "image_path": image_info["path"]
            }
            
            # 3. 벡터 데이터 구성
            dense_vector = {
                "id": f"{document_id}_{self.language}_{image_info['chunk_id']}_image_dense",
                "values": dense_embedding,
                "metadata": metadata
            }
            sparse_vector = {
                "id": f"{document_id}_{self.language}_{image_info['chunk_id']}_image_sparse",
                "sparse_values": {
                    "indices": sparse_embedding['indices'],
                    "values": sparse_embedding['values']
                },
                "metadata": metadata
            }
            
            # 4. Dense/Sparse 벡터 병렬 적재
            dense_task = asyncio.create_task(self.upsert_dense_vector(dense_vector, document_id))
            sparse_task = asyncio.create_task(self.upsert_sparse_vector(document_id, sparse_vector, metadata))
            
            await dense_task
            await sparse_task
            
            return True
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return False

    async def process_images(self, image_infos: List[Dict[str, Any]], document_id: str) -> bool:
        try:
            # 1. BM25 인코더 초기화
            self.bm25 = BM25Encoder()
            self.bm25.fit([info["summary"] for info in image_infos])
            
            # 2. 이미지별 파이프라인 처리
            tasks = []
            for i, image_info in enumerate(image_infos):
                image_info["chunk_id"] = i + 1
                task = asyncio.create_task(
                    self.process_image(image_info, document_id)
                )
                tasks.append(task)
            
            # 3. 모든 이미지 처리 완료 대기
            results = await asyncio.gather(*tasks)
            return all(results)
            
        except Exception as e:
            logger.error(f"Images processing error: {e}")
            return False

    # 처리된 벡터 통계 반환
    def get_processing_stats(self) -> Dict[str, Any]:
    
        return {
            "text": self.processed_vectors["text"],
            "image": self.processed_vectors["image"]
        }