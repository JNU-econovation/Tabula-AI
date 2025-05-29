from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, Form, Depends, File, HTTPException
from bson import ObjectId
import tempfile
import os
from pathlib import Path

from result_service.service import ResultService
from common_sdk import get_logger
from common_sdk.auth import get_current_member
from common_sdk.crud.s3 import S3Storage
from common_sdk.crud.mongodb import MongoDB
from common_sdk.swagger import result_service_response, result_service_space_response

from common_sdk.exceptions import (
    MissingResultFileData, UnsupportedResultFileFormat
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
ALLOWED_FILE_TYPES = ["application/pdf", "image/png"]

@router.post("/{space_id}/result", responses=result_service_response)
async def upload_result(
    space_id: str,
    files: List[UploadFile] = File(...),
    userId: int = Depends(get_current_member)
) -> Dict[str, Any]:
    
    # 파일 데이터 누락 검증
    if not files or len(files) == 0:
        logger.error(f"User: {userId} - File data is missing")
        raise MissingResultFileData()
    
    # 파일 개수 및 형식 검증
    if len(files) == 1:
        # 단일 파일인 경우: PDF 또는 PNG 허용
        file = files[0]
        if file.content_type not in ALLOWED_FILE_TYPES:
            logger.error(f"User: {userId} - Unsupported file type: {file.content_type}")
            raise UnsupportedResultFileFormat()
    else:
        # 여러 파일인 경우: 모두 PNG여야 함
        for i, file in enumerate(files):
            if file.content_type != "image/png":
                logger.error(f"User: {userId} - Multiple files must be PNG. File {i+1}: {file.content_type}")
                raise UnsupportedResultFileFormat()
    
    try:
        # result_id 생성
        result_id = str(ObjectId())

        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.gettempdir()) / "result_service" / result_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 파일들 저장
        saved_files = []
        
        for i, file in enumerate(files):
            if not file.filename:
                raise HTTPException(status_code=400, detail=f"파일 {i+1}의 이름이 없습니다.")
            
            # 파일 내용 읽기
            content = await file.read()
            
            # 파일 저장
            if len(files) == 1:
                # 단일 파일인 경우 원본 이름 유지
                file_path = temp_dir / file.filename
            else:
                # 여러 파일인 경우 순서대로 이름 지정
                file_extension = Path(file.filename).suffix
                file_path = temp_dir / f"page_{i+1:03d}{file_extension}"
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            saved_files.append(str(file_path))
            logger.info(f"User: {userId} - File {i+1} saved: {file.filename}")
        
        logger.info(f"Space: {space_id} - All files uploaded successfully. Total files: {len(files)}")
        
        # 여러 파일인 경우 첫 번째 파일명을 대표 이름으로 사용
        representative_filename = files[0].filename if len(files) == 1 else f"{len(files)}개_이미지파일"
        
        # ResultService 인스턴스 생성
        service = ResultService(
            result_id=result_id,
            file_paths=saved_files,
            file_name=representative_filename,
            user_id=userId,
            space_id=space_id
        )
        
        # 채점 처리 실행
        logger.info(f"Space: {space_id} - Starting grading process for result: {result_id}")
        processing_result = await service.process_grading()
        logger.info(f"Space: {space_id} - Grading process completed for result: {result_id}")

        response = {
            "success": True,
            "response": {
                "resultId": result_id,
                "fileName": representative_filename
            },
            "error": None
        }

        return response
    
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Space: {space_id} - Error in upload_result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

@router.get("/{space_id}/progress/{resultId}", responses=result_service_space_response)
async def get_progress():
    pass