from bson import ObjectId
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, Depends, File

from note_service.service import NoteService
from common_sdk.sse import get_progress_stream
from common_sdk.auth import get_current_member
from common_sdk.crud.s3 import S3Storage
from common_sdk.crud.mongodb import MongoDB
from common_sdk.sse import update_progress, get_progress_stream
from note_sdk.config import settings
from common_sdk.swagger import note_service_response, note_service_space_response
from common_sdk.exceptions import (
    UnsupportedNoteFileFormat, NoteFileSizeExceeded, SpaceIdNotFound, 
    MissingSpaceId, MissingNoteFileData, MissingFieldData
)

from common_sdk.get_logger import get_logger

# 로그 설정
logger = get_logger()


router = APIRouter(
    tags=["Note Service"]
)

# 인스턴스 생성
service_instances: Dict[str, NoteService] = {}
s3_storage = S3Storage()
mongodb = MongoDB()

# 지원하는 파일 형식
ALLOWED_FILE_TYPES = ["application/pdf"]

# 학습 자료 업로드 API
@router.post("/{folderId}/upload", responses=note_service_response)
async def upload_file(
    folderId: str,
    userId: int = Depends(get_current_member),
    file: UploadFile = File(None),
    langType: str = Form(None),
    fileDomain: str = Form(None)
) -> Dict[str, Any]:
    
    # 파일 데이터 누락 검증
    if not file:
        logger.error(f"User: {userId} - File data is missing")
        raise MissingNoteFileData()

    # 필드 데이터 누락 검증
    if not langType or not fileDomain:
        logger.error(f"User: {userId} - Field data is missing langType: {langType}, fileDomain: {fileDomain}")
        raise MissingFieldData()

    # 파일 형식 검증
    if file.content_type not in ALLOWED_FILE_TYPES:
        logger.error(f"User: {userId} - Unsupported file type: {file.content_type}")
        raise UnsupportedNoteFileFormat()
    
    # 파일 용량 검증
    if file.size > 5 * 1024 * 1024:  # 5MB
        logger.error(f"User: {userId} - File size exceeds limit: {file.size}")
        raise NoteFileSizeExceeded()

    try:
        # space_id 생성
        space_id = str(ObjectId())
        
        # 초기 상태 설정
        update_progress(space_id, 0, {
            "status": "파일 업로드 준비 중",
            "result": {"spaceId": space_id}
        })
        
        # space_id 기반 작업 디렉토리 생성
        settings.ensure_user_directories(space_id)
        
        # 원본 파일 경로
        origin_dir = settings.get_origin_dir(space_id)
        origin_path = origin_dir / file.filename

        # 파일 저장
        with open(origin_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 파일 업로드(S3)
        s3_result = s3_storage.upload_note_file(
            file_path=str(origin_path),
            user_id=userId,
            space_id=space_id
        )
        
        # 서비스 인스턴스 생성 및 저장
        service = NoteService(
            pdf=str(origin_path),
            folder_id=folderId,
            language=langType,
            domain_type=fileDomain,
            space_id=space_id,
            s3_url=s3_result,
            file_name=file.filename,
            user_id=userId
        )
        service_instances[space_id] = service

        response = {
            "success": True,
            "response": {
                "spaceId": space_id,
            },
            "error": None
        }

        return response
        
    except Exception as e:
        if space_id:
            update_progress(space_id, -1, {
                "status": f"에러 발생: {str(e)}",
                "result": {"spaceId": space_id}
            })
        raise


# 학습 자료 진행률 조회 및 결과 확인 API
@router.get("/{folderId}/progress/{spaceId}", responses=note_service_space_response)
async def get_progress(
    folderId: str,
    spaceId: str,
    user_id: int = Depends(get_current_member)
):
    """
    SSE를 사용하여 진행률을 실시간으로 전송
    """
    try:
        # spaceId 누락 체크
        if not spaceId:
            logger.error(f"User: {user_id} - spaceId is missing")
            raise MissingSpaceId()

        # 서비스 인스턴스 조회
        service = service_instances.get(spaceId)
        if not service:
            logger.error(f"User: {user_id} - Space not found: {spaceId}")
            raise SpaceIdNotFound()
            
        # SSE 연결 시작
        return get_progress_stream(spaceId, service)
        
    except Exception as e:
        logger.error(f"User: {user_id} - Error in get_progress: {str(e)}")
        update_progress(spaceId, -1, {
            "status": f"진행률 조회 실패: {str(e)}",
            "result": {"spaceId": spaceId}
        })
        raise