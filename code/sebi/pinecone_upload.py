import os
import uuid
from typing import List
from fastapi import File, UploadFile, APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
import tempfile
import shutil

# 환경변수 로드
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

router = APIRouter(
    tags=['AI']
)

# Pinecone 초기화 및 인덱스 확인 함수
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "tabular"
    
    # 인덱스가 없으면 생성
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
    
    return pc.Index(index_name)

# 문서 처리 및 업로드 함수
async def process_and_upload_document(file_path, document_id):
    try:
        # 인덱스 연결
        index = init_pinecone()
        
        # 문서 불러오기
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        print(f"문서 수: {len(docs)}")
        
        # 문서 청크 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        print(f"분할된 청크 수: {len(split_docs)}")
        
        # 임베딩 모델 준비
        dense_embedder = UpstageEmbeddings(
            model="solar-embedding-1-large-passage",
            upstage_api_key=UPSTAGE_API_KEY
        )
        
        texts = [doc.page_content for doc in split_docs]
        
        # Sparse 인코더 준비 (BM25)
        bm25 = BM25Encoder()
        bm25.fit(texts)
        
        # Dense 임베딩 생성
        dense_vectors = dense_embedder.embed_documents(texts)
        
        # 업서트 (배치로)
        batch_size = 100
        uploaded_count = 0
        
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            # 각 청크마다 고유한 ID 생성
            batch_ids = [f"{document_id}_{j+1}" for j in range(i, end)]  # document_id와 chunk_id를 조합한 ID
            batch_texts = texts[i:end]
            sparse_vecs = bm25.encode_documents(batch_texts)
            
            batch = []
            for j in range(end - i):
                chunk_id = i + j + 1  # 청크 ID는 1부터 시작
                batch.append({
                    "id": batch_ids[j],
                    "values": dense_vectors[i + j],  # dense embedding 벡터
                    "sparse_values": sparse_vecs[j],  # sparse 벡터
                    "metadata": {
                        "document_id": document_id,  # 문서 식별자
                        "chunk_id": chunk_id,        # 청크 식별자
                        "type": "text",              # 타입 구분
                        "content": batch_texts[j],   # 원본 텍스트
                        # 필요하다면 이미지 참조 정보를 추가할 수 있음
                        # "image_references": []
                    }
                })
            
            # 인덱스에 하이브리드 벡터 저장 (namespace 사용하지 않음)
            index.upsert(vectors=batch)
            uploaded_count += len(batch)
            print(f"{i+1}~{end} 업로드 완료")
        
        return {"message": "하이브리드 인덱스 저장 완료", "chunks_uploaded": uploaded_count, "document_id": document_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류 발생: {str(e)}")

# API 엔드포인트 설정
@router.post("/upload", response_class=JSONResponse)
async def upload_study_document(file: UploadFile = File(...)):
    # 고유한 문서 ID 생성
    document_id = str(uuid.uuid4())
    
    # 파일 확장자 검사
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="마크다운(.md) 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    try:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        # 문서 처리 및 업로드
        result = await process_and_upload_document(temp_file.name, document_id)
        return {"document_id": document_id, **result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")
    finally:
        os.unlink(temp_file.name)  # 임시 파일 삭제