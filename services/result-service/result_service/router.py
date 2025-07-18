# result_service/router.py
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, Depends, File, HTTPException
from bson import ObjectId
import tempfile
from pathlib import Path
import fitz

from result_service.service import ResultService
from common_sdk import get_logger
from common_sdk.auth import get_current_member
from common_sdk.crud.s3 import S3Storage
from common_sdk.crud.mongodb import MongoDB
from common_sdk.swagger import result_service_response, result_service_space_response
from common_sdk.sse import update_result_progress, get_result_progress_stream
from common_sdk.exceptions import (
    MissingResultFileData, UnsupportedResultFileFormat,
    MissingResultId, ResultIdNotFound, ResultFileUploadPageExceeded
)


# 로그 설정
logger = get_logger()

router = APIRouter(
    tags=["Result Service"]
)

# 전역 인스턴스들
service_instances: Dict[str, ResultService] = {}
s3_storage = S3Storage()
mongodb = MongoDB()

# 지원하는 파일 형식
ALLOWED_FILE_TYPES = ["application/pdf", "image/png", "image/jpeg", "image/jpg"]
MAX_PDF_PAGES = 6

@router.post("/{spaceId}/result", responses=result_service_response)
async def upload_result(
    spaceId: str,
    file: List[UploadFile] = File(...),
    userId: int = Depends(get_current_member)
) -> Dict[str, Any]:
    
    # 파일 데이터 누락 검증
    if not file or len(file) == 0:
        logger.error(f"User: {userId} - File data is missing")
        raise MissingResultFileData()
    
    # 파일 개수 및 형식 검증
    if len(file) == 1:
        # 단일 파일인 경우: PDF 또는 IMAGE 허용
        single_file = file[0]
        if single_file.content_type not in ALLOWED_FILE_TYPES:
            logger.error(f"User: {userId} - Unsupported file type: {single_file.content_type}")
            raise UnsupportedResultFileFormat()
    else:
        # 여러 파일인 경우: 모두 PNG여야 함
        for i, single_file in enumerate(file):
            if single_file.content_type == "application/pdf":
                logger.error(f"User: {userId} - Multiple files must be IMAGE. File {i+1}: {single_file.content_type}")
                raise UnsupportedResultFileFormat()
    
    result_id = None
    try:
        # result_id 생성
        result_id = str(ObjectId())

        # 초기 상태 설정 (Result Service용 함수 사용)
        update_result_progress(result_id, 0, {
            "resultId": result_id,
            # "status": "processing"
        })

        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.gettempdir()) / "result_service" / result_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 파일들 저장
        saved_files = []
        
        for i, single_file in enumerate(file):
            if not single_file.filename:
                raise HTTPException(status_code=400, detail=f"파일 {i+1}의 이름이 없습니다.")
            
            # 파일 내용 읽기
            content = await single_file.read()

            # PDF 파일인 경우 페이지 수 체크
            if single_file.content_type == "application/pdf":
                check_pdf_page_count(content, single_file.filename, userId)
            
            # 파일 저장
            if len(file) == 1:
                # 단일 파일인 경우 원본 이름 유지
                file_path = temp_dir / single_file.filename
            else:
                # 여러 파일인 경우 순서대로 이름 지정
                file_extension = Path(single_file.filename).suffix
                file_path = temp_dir / f"page_{i+1:03d}{file_extension}"
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            saved_files.append(str(file_path))
            logger.info(f"User: {userId} - File {i+1} saved: {single_file.filename}")
        
        logger.info(f"Space: {spaceId} - All files uploaded successfully. Total files: {len(file)}")
        
        # 여러 파일인 경우 첫 번째 파일명을 대표 이름으로 사용
        representative_filename = file[0].filename if len(file) == 1 else f"{len(file)}개_이미지파일"
        
        # ResultService 인스턴스 생성 및 저장
        service = ResultService(
            result_id=result_id,
            file_paths=saved_files,
            file_name=representative_filename,
            user_id=userId,
            space_id=spaceId
        )
        
        # 서비스 인스턴스 저장 (SSE용)
        service_instances[result_id] = service

        # 업로드 완료 후 진행률 초기화 (SSE 연결 전 미리 설정)
        update_result_progress(result_id, 0, {
            "resultId": result_id,
            # "status": "processing"
        })

        response = {
            "success": True,
            "response": {
                "resultId": result_id
            },
            "error": None
        }

        return response
    
    except Exception as e:
        if result_id:
            update_result_progress(result_id, -1, {
                "status": f"에러 발생: {str(e)}",
                "result": {"resultId": result_id}
            })
        raise

# 진행률 확인
@router.get("/{spaceId}/progress/{resultId}", responses=result_service_space_response)
async def get_progress(
    spaceId: str,
    resultId: str,
    user_id: int = Depends(get_current_member)
):
    try:
        # resultId 누락 체크
        if not resultId:
            logger.error(f"User: {user_id} - resultId is missing")
            raise MissingResultId()

        # 서비스 인스턴스 조회
        service = service_instances.get(resultId)
        if not service:
            logger.error(f"User: {user_id} - Result not found: {resultId}")
            raise ResultIdNotFound()
            
        # SSE 연결 시작
        return get_result_progress_stream(resultId, service)
        
    except Exception as e:
        logger.error(f"User: {user_id} - Error in get_progress: {str(e)}")
        update_result_progress(resultId, -1, {
            "resultId": resultId,
            "error": {str(e)}
        })
        raise

def check_pdf_page_count(file_content: bytes, filename: str, user_id: int) -> None:
    """PyMuPDF를 사용하여 PDF 파일의 페이지 수를 체크하는 함수"""
    doc = None
    try:
        # PyMuPDF로 PDF 문서 열기
        doc = fitz.open(stream=file_content, filetype="pdf")
        page_count = len(doc)
        
        if page_count > MAX_PDF_PAGES:
            logger.error(f"User: {user_id} - PDF page count exceeded: {filename} has {page_count} pages (max: {MAX_PDF_PAGES})")
            raise ResultFileUploadPageExceeded()
        
        logger.info(f"User: {user_id} - PDF page count check passed: {filename} has {page_count} pages")
        
    except ResultFileUploadPageExceeded:
        raise
    except Exception as e:
        logger.error(f"User: {user_id} - Error checking PDF page count for {filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF 파일 페이지 수 확인 중 오류가 발생했습니다: {str(e)}")
    finally:
        # 리소스 해제
        if doc is not None:
            doc.close()