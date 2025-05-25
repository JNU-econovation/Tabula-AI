import json
import asyncio
from typing import Dict
from sse_starlette.sse import EventSourceResponse

# 진행률 저장소
progress_store: Dict[str, int] = {}

def update_progress(task_id: str, progress: int):
    """진행률 업데이트"""
    progress_store[task_id] = progress

async def progress_generator(task_id: str):
    """SSE 이벤트 생성기"""
    while True:
        if task_id in progress_store:
            progress = progress_store[task_id]
            if progress == 100 or progress == -1:
                # 완료 시 결과물 정보 포함
                result = {
                    "progress": progress,
                    "status": "success" if progress == 100 else "error",
                    "result": {
                        "document_id": task_id,
                        "processing_time": progress_store.get(f"{task_id}_time", 0),
                        "markdown_path": f"/api/v1/spaces/{task_id}/markdown",
                        "keyword_path": f"/api/v1/spaces/{task_id}/keyword"
                    }
                }
                yield {
                    "event": "complete",
                    "data": json.dumps(result)
                }
                break
            yield {
                "event": "progress",
                "data": json.dumps({
                    "progress": progress,
                    "status": "processing"
                })
            }
        await asyncio.sleep(0.1)

def get_progress_stream(task_id: str):
    """진행률 스트림 반환"""
    return EventSourceResponse(progress_generator(task_id))