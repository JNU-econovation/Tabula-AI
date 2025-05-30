import re
import asyncio
import json
from pinecone import Pinecone
from typing import List, Dict, Any, Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from pinecone_text.sparse import BM25Encoder

from common_sdk.exceptions import ExternalConnectionError
from common_sdk.utils import get_embedding
from common_sdk.config import settings as common_settings
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()


# 벡터 로더 클래스
class VectorLoader:
    def __init__(self, language: Literal["ko", "en"], space_id: str = None):

        self.language = language
        self.space_id = space_id
        
        # Pinecone 클라이언트 초기화
        self.pc = Pinecone(api_key=common_settings.PINECONE_API_KEY)
        
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

    def init_dense_index(self) -> Any:
        try:
            # 언어에 따른 인덱스 설정
            index_name = common_settings.INDEX_NAME_KOR_DEN_CONTENTS if self.language == "ko" else common_settings.INDEX_NAME_ENG_DEN_CONTENTS
            
            # 언어별 차원 설정
            dimension = 4096 if self.language == "ko" else 3072  # Upstage: 4096, OpenAI 3-large: 3072
            
            # 인덱스 존재 확인
            if index_name not in self.pc.list_indexes().names():
                logger.error(f"[VectorLoader] '{index_name}' dense index not found")
                raise ExternalConnectionError()
            
            return self.pc.Index(index_name)
        
        except Exception as e:
            logger.error(f"[VectorLoader] Dense index initialization error: {e}")
            raise ExternalConnectionError()

    def init_sparse_index(self) -> Any:
        try:
            # 언어에 따른 인덱스 설정
            index_name = common_settings.INDEX_NAME_KOR_SPA_CONTENTS if self.language == "ko" else common_settings.INDEX_NAME_ENG_SPA_CONTENTS
            
            # 인덱스 존재 확인
            if index_name not in self.pc.list_indexes().names():
                logger.error(f"[VectorLoader] '{index_name}' sparse index not found")
                raise ExternalConnectionError()
            
            return self.pc.Index(index_name)
        
        except Exception as e:
            logger.error(f"[VectorLoader] Sparse index initialization error: {e}")
            raise ExternalConnectionError()


    def extract_image_paths(self, content: str) -> List[Dict[str, str]]:
        pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.finditer(pattern, content)
        imagePaths = [{"imagePath": m.group(2)} for m in matches]
        
        if not imagePaths:
            logger.info(f"[VectorLoader] No image paths found in content")
            return []
        
        return imagePaths

    def find_images_in_chunk(self, chunk: str, imagePaths: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not imagePaths:
            return []
        return [img for img in imagePaths if img["imagePath"] in chunk]

    async def process_chunk(self, chunk: str, chunkId: int, imagePaths: List[Dict[str, str]]) -> bool:
        try:
            # 1. 청크에 포함된 이미지 경로 찾기
            chunk_imagePaths = self.find_images_in_chunk(chunk, imagePaths)
            
            # 2. Dense 임베딩 생성
            dense_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                get_embedding,
                chunk,
                self.language
            )
            
            # Dense 임베딩 검증
            if not dense_embedding or len(dense_embedding) == 0:
                logger.error(f"Dense embedding generation failed for chunk {chunkId}")
                return False
            
            # 3. Sparse 임베딩 생성 및 검증
            sparse_embedding = None
            try:
                sparse_result = self.bm25.encode_documents([chunk])
                if sparse_result and len(sparse_result) > 0:
                    sparse_embedding = sparse_result[0]
                    
                    if not sparse_embedding or 'indices' not in sparse_embedding or 'values' not in sparse_embedding:
                        logger.warning(f"[VectorLoader] Invalid sparse embedding structure for chunk {chunkId}")
                        sparse_embedding = None
                    elif len(sparse_embedding['indices']) == 0 or len(sparse_embedding['values']) == 0:
                        logger.warning(f"[VectorLoader] Empty sparse embedding for chunk {chunkId}")
                        sparse_embedding = None
                    
            except Exception as e:
                logger.error(f"[VectorLoader] Sparse embedding error for chunk {chunkId}: {e}")
                sparse_embedding = None
            
            # 4. 메타데이터 구성
            image_references_json = json.dumps([{"imagePath": str(img["imagePath"])} for img in chunk_imagePaths]) if chunk_imagePaths else "[]"
            
            metadata = {
                "spaceId": str(self.space_id),
                "chunkId": int(chunkId),
                "type": "text",
                "content": str(chunk),
                "imageReferences": image_references_json
            }
            
            # 5. Dense 벡터 데이터 구성 및 업로드
            vector_id = f"{str(self.space_id)}_{self.language}_{chunkId}_text"
            dense_vector = {
                "id": f"{vector_id}_dense",
                "values": dense_embedding,
                "metadata": metadata
            }

            dense_task = asyncio.create_task(self.upsert_dense_vector(dense_vector))
            
            # 6. Sparse 벡터 처리 (성공한 경우에만)
            sparse_task = None
            if sparse_embedding:
                sparse_vector = {
                    "id": f"{vector_id}_sparse",
                    "sparse_values": {
                        "indices": sparse_embedding['indices'],
                        "values": sparse_embedding['values']
                    },
                    "metadata": metadata
                }
                sparse_task = asyncio.create_task(self.upsert_sparse_vector(sparse_vector))
            else:
                logger.info(f"[VectorLoader] Skipping sparse vector for chunk {chunkId} due to empty embedding")
            
            # 7. 병렬 실행
            if sparse_task:
                await asyncio.gather(dense_task, sparse_task)
            else:
                await dense_task
            
            return True
            
        except Exception as e:
            logger.error(f"[VectorLoader] Chunk processing error: {e}")
            return False

    async def upsert_dense_vector(self, vector: Dict[str, Any]) -> bool:
        try:
            self.dense_index.upsert(vectors=[vector], namespace="documents")
            self.processed_vectors["text"]["dense"] += 1
            return True
        except Exception as e:
            logger.error(f"[VectorLoader] Dense vector upsertion error: {e}")
            return False

    async def upsert_sparse_vector(self, sparse_vector: Dict) -> bool:
        try:
            # Sparse 벡터 데이터 검증
            sparse_values = sparse_vector.get('sparse_values', {})
            indices = sparse_values.get('indices', [])
            values = sparse_values.get('values', [])
            
            if not indices or not values or len(indices) != len(values):
                logger.warning(f"[VectorLoader] Invalid sparse vector data: indices={len(indices)}, values={len(values)}")
                return False
            
            # Pinecone에 업서트
            self.sparse_index.upsert(
                vectors=[{
                    'id': sparse_vector['id'],
                    'sparse_values': sparse_vector['sparse_values'],
                    'metadata': sparse_vector['metadata']
                }],
                namespace='documents'
            )
            self.processed_vectors["text"]["sparse"] += 1
            return True
        except Exception as e:
            logger.error(f"[VectorLoader] Sparse vector upsertion error: {str(e)}")
            return False

    async def process_markdown(self, content: str, document_id: str) -> bool:
        try:
            # 1. 이미지 경로 추출
            imagePaths = self.extract_image_paths(content)
            
            # 2. 텍스트 분할
            chunks = self.text_splitter.split_text(content)
            
            if not chunks:
                logger.warning("No chunks found in content")
                return False
            
            # 3. BM25 인코더 초기화
            self.bm25 = BM25Encoder()
            
            # 빈 청크 필터링
            non_empty_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            if not non_empty_chunks:
                logger.error(f"[VectorLoader] All chunks are empty after filtering")
                return False
            
            try:
                self.bm25.fit(non_empty_chunks)
            except Exception as e:
                logger.error(f"[VectorLoader] BM25 encoder fitting failed: {e}")
                return False
            
            # 4. 청크별 파이프라인 처리
            tasks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 빈 청크 스킵
                    task = asyncio.create_task(
                        self.process_chunk(chunk, i + 1, imagePaths)
                    )
                    tasks.append(task)
                else:
                    logger.info(f"[VectorLoader] Skipping empty chunk {i + 1}")
            
            if not tasks:
                logger.error(f"[VectorLoader] No valid chunks to process")
                return False
            
            # 5. 모든 청크 처리 완료 대기
            results = await asyncio.gather(*tasks)
            success_count = sum(results)
            
            logger.info(f"[VectorLoader] Processed {success_count}/{len(tasks)} chunks successfully")
            
            return all(results)
            
        except Exception as e:
            logger.error(f"[VectorLoader] Markdown processing error: {e}")
            return False