import json
import asyncio
from typing import Dict, Any
from sse_starlette.sse import EventSourceResponse

# 진행률 저장소
progress_store: Dict[str, Dict[str, Any]] = {}

def update_progress(task_id: str, progress: int, result: Dict[str, Any] = None):
    """진행률 업데이트"""
    if task_id not in progress_store:
        progress_store[task_id] = {"progress": 0, "result": None}
    
    progress_store[task_id]["progress"] = progress
    if result:
        progress_store[task_id]["result"] = result

async def progress_generator(task_id: str):
    """SSE 이벤트 생성기"""
    while True:
        if task_id in progress_store:
            data = progress_store[task_id]
            progress = data["progress"]
            
            if progress == 100:
                # 완료 시 결과물 정보 포함
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "success": True,
                        "response": {
                            "progress": 100,
                            "spaceId": data["result"].get("spaceId"),
                            "spaceName": data["result"].get("spaceName"),
                            "keywords": data["result"].get("keywords")
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
                        "error": "SSE Error"
                    })
                }
                break
            else:
                # 진행 중인 경우
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "success": True,
                        "response": {
                            "progress": progress,
                            "status": "processing"
                        },
                        "error": None
                    })
                }
        await asyncio.sleep(0.1)

def get_progress_stream(task_id: str):
    """진행률 스트림 반환"""
    return EventSourceResponse(progress_generator(task_id))