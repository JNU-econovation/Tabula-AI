import json
import asyncio

from typing import Dict, Any
from sse_starlette.sse import EventSourceResponse

from common_sdk.get_logger import get_logger
from common_sdk.constants import ProgressPhase, ProgressRange, StatusMessage

logger = get_logger()

# 진행률 저장소
progress_store: Dict[str, Dict[str, Any]] = {}

# 진행률 업데이트
def update_progress(space_id: str, progress: int, status: Dict[str, Any] = None):
    if space_id not in progress_store:
        progress_store[space_id] = {
            "progress": 0,
            "status": {"status": StatusMessage.INITIAL},
            "result": {"spaceId": space_id}
        }
    
    progress_store[space_id]["progress"] = progress
    if status:
        progress_store[space_id]["status"] = status
        if "result" in status:
            progress_store[space_id]["result"] = status["result"]

async def progress_generator(space_id: str, service: Any = None):
    try:
        # 1. 초기 상태 확인
        if space_id not in progress_store:
            update_progress(space_id, 0, {
                "status": StatusMessage.INITIAL,
                "result": {"spaceId": space_id}
            })
        
        # 2. 처리 시작
        if service:
            # 비동기 처리 시작
            asyncio.create_task(service.process_document())
        
        # 3. SSE 연결 시작
        while True:
            if space_id in progress_store:
                data = progress_store[space_id]
                progress = data["progress"]
                status = data.get("status", {})
                result = data.get("result", {})
                
                # 4. 진행 상태에 따른 이벤트 전송
                if progress == 100:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "success": True,
                            "response": {
                                "progress": 100,
                                "spaceId": result.get("spaceId"),
                                "spaceName": result.get("spaceName"),
                            },
                            "error": None
                        })
                    }
                    break
                elif progress == -1:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "success": False,
                            "response": None,
                            "error": status.get("status", StatusMessage.ERROR)
                        })
                    }
                    break
                else:
                    # 진행 중인 경우
                    status_message = status.get("status", "처리 중")
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "success": True,
                            "response": {
                                "progress": progress,
                                "status": status_message
                            },
                            "error": None
                        })
                    }
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"[progress_generator] SSE Error: {str(e)}")
        yield {
            "event": "error",
            "data": json.dumps({
                "success": False,
                "response": None,
                "error": f"SSE Error: {str(e)}"
            })
        }

def get_progress_stream(space_id: str, service: Any = None):
    """진행률 스트림 반환"""
    return EventSourceResponse(progress_generator(space_id, service))