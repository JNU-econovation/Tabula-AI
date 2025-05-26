import uuid

from typing import Dict, Any
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, Depends, HTTPException

from note_service.service import NoteService
from common_sdk.sse import get_progress_stream
from common_sdk.auth.decoded_token import get_current_member
from note_sdk.config import settings
from common_sdk.swagger import note_service_response, note_service_task_response

from common_sdk.get_logger import get_logger

# 로그 설정
logger = get_logger()


router = APIRouter(
    tags=["Note Service"]
)

# 서비스 인스턴스 저장소
service_instances: Dict[str, NoteService] = {}

# 학습 자료 업로드 API
@router.post("/{folderId}/upload", responses=note_service_response)
async def upload_file(
    folderId: int,
    background_tasks: BackgroundTasks,
    userId: int = Depends(get_current_member),
    file: UploadFile = Form(...),
    langType: str = Form(...),
    fileDomain: str = Form(...)
) -> Dict[str, Any]:
    """
    1. 토큰 디코딩 진행 필요
    2. 파일 확장자 제한(pdf)
    3. 파일 용량 제한(5MB) 필요 
    """

    # # 지원하는 파일 형식
    # ALLOWED_FILE_TYPES = ["image/jpeg", "image/png"]

    # # 파일 형식 검증
    # if file.content_type not in ALLOWED_FILE_TYPES:
    #     logger.error(f"userId: {userId} Unsupported file type: {file.content_type}")
    #     raise HTTPException(status_code=400, detail="Unsupported file type")

    # 파일 업로드 API
    task_id = str(uuid.uuid4())
    
    # task_id 기반 작업 디렉토리 생성
    settings.ensure_user_directories(task_id)
    
    # 원본 파일 경로
    origin_dir = settings.get_origin_dir(task_id)
    origin_path = origin_dir / file.filename

    """
    4. 원본 파일 저장(S3) 로직 필요: 반환값인 url 가지고있기
    """

    # 파일 저장
    with open(origin_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # 서비스 인스턴스 생성 및 저장
    service = NoteService(
        pdf=str(origin_path),
        language=langType,
        domain_type=fileDomain,
        task_id=task_id,
    )
    service_instances[task_id] = service
    
    # 백그라운드 처리 시작
    background_tasks.add_task(service.process_document)

    response = {
        "success": True,
        "response": {
            "taskId": task_id
        },
        "error": None
    }

    return response


# 학습 자료 진행률 조회 및 결과 확인 API
@router.get("/{folderId}/progress/{taskId}", responses=note_service_task_response)
async def get_progress(
    folderId: int,
    taskId: str,
    user_id: int = Depends(get_current_member)
):
    """
    SSE를 사용하여 진행률을 실시간으로 전송
    """
    # 서비스 인스턴스 조회
    service = service_instances.get(taskId)
    if not service:
        logger.error(f"userId: {user_id} Task not found: {taskId}")
        raise HTTPException(status_code=404, detail="Task not found")

    # 진행률 스트림 반환
    return get_progress_stream(taskId)