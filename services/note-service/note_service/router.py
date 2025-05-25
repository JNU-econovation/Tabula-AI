import uuid
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, Depends, HTTPException

from note_service.service import NoteService
from common_sdk.sse import get_progress_stream
from common_sdk.auth.decoded_token import get_current_member
from note_sdk.config import settings

router = APIRouter(
    tags=["Note Service"]
)

# 서비스 인스턴스 저장소
service_instances: Dict[str, NoteService] = {}

# 학습 자료 업로드 API
@router.post("/{folderId}/upload")
async def upload_file(
    folderId: int,
    background_tasks: BackgroundTasks,
    # userId: int = Depends(get_current_member),
    file: UploadFile = Form(...),
    langType: str = Form(...),
    fileDomain: str = Form(...)
) -> Dict[str, Any]:
    """
    1. 토큰 디코딩 진행 필요
    2. 파일 확장자 제한(pdf)
    """
    # 파일 업로드 API
    task_id = str(uuid.uuid4())
    
    # task_id 기반 작업 디렉토리 생성
    settings.ensure_user_directories(task_id)
    
    # 원본 파일 경로
    origin_dir = settings.get_origin_dir(task_id)
    origin_path = origin_dir / file.filename

    """
    3. 파일 용량 제한(5MB) 필요
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
@router.get("/{folderId}/progress/{taskId}")
async def get_progress(
    folderId: int,
    taskId: str,
    # user_id: int = Depends(get_current_member)
) -> Dict[str, Any]:
    
    try:
        # 서비스 인스턴스 조회
        service = service_instances.get(taskId)
        if not service:
            raise HTTPException(status_code=404, detail="Task not found")

        # 현재 진행률 조회
        current_progress = service.get_current_progress()
        
        if current_progress == 100:
            # 완료된 경우 키워드 결과 포함
            response = {
                "success": True,
                "response": {
                    "progress": 100,
                    "spaceId": folderId,
                    "spaceName": service.document_id,
                    "keywords": service.get_keyword_result()
                },
                "error": None
            }

            return response
        
        else:
            # 진행 중인 경우
            response = {
                "success": True,
                "response": {
                    "progress": current_progress,
                    "status": "processing"
                },
                "error": None
            }

            return response
            
    except Exception as e:
        response = {
            "success": False,
            "response": None,
            "error": str(e)
        }

        return response