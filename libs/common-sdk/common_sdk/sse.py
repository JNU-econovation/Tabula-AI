import json
import asyncio

from typing import Dict, Any
from sse_starlette.sse import EventSourceResponse

from common_sdk.get_logger import get_logger

logger = get_logger()

# 진행률 저장소
progress_store: Dict[str, Dict[str, Any]] = {}

# 진행률 업데이트
def update_progress(space_id: str, progress: int, status: Dict[str, Any] = None):
    if space_id not in progress_store:
        progress_store[space_id] = {
            "progress": 0,
            "status": {"status": "처리 시작"},
            "result": {"spaceId": space_id}
        }
    
    progress_store[space_id]["progress"] = progress
    if status:
        progress_store[space_id]["status"] = status
        if "result" in status:
            progress_store[space_id]["result"] = status["result"]

async def progress_generator(space_id: str):
    # 초기 상태 설정
    if space_id not in progress_store:
        progress_store[space_id] = {
            "progress": 0,
            "status": {"status": "처리 시작"},
            "result": {"spaceId": space_id}
        }
    
    try:
        while True:
            if space_id in progress_store:
                data = progress_store[space_id]
                progress = data["progress"]
                status = data.get("status", {})
                result = data.get("result", {})
                
                if progress == 100:
                    # 완료 시 결과물 정보 포함
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
                    # 에러 발생
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "success": False,
                            "response": None,
                            "error": status.get("status", "SSE Error")
                        })
                    }
                    break
                else:
                    # 진행 중인 경우
                    status_message = status.get("status", "processing")
                    if progress <= 30:
                        phase = "PDF 파싱"
                    elif progress <= 60:
                        phase = "마크다운 처리"
                    elif progress <= 90:
                        phase = "키워드 생성"
                    else:
                        phase = "MongoDB 적재"
                    
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "success": True,
                            "response": {
                                "progress": progress,
                                "phase": phase,
                                "status": status_message,
                                "spaceId": result.get("spaceId")
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

def get_progress_stream(space_id: str):
    """진행률 스트림 반환"""
    return EventSourceResponse(progress_generator(space_id))