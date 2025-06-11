import json
import asyncio

from typing import Dict, Any
from sse_starlette.sse import EventSourceResponse

from common_sdk.get_logger import get_logger
from common_sdk.constants import StatusMessage

logger = get_logger()

# Note Service

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
                                "spaceName": result.get("spaceName")
                            },
                            "error": None
                        })
                    }
                    break
                elif progress == -1:
                    error_message = status.get("status", StatusMessage.ERROR)
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "success": False,
                            "response": None,
                            "error": error_message
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
                                "progress": progress
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

# def get_progress_stream(space_id: str, service: Any = None):
#     """진행률 스트림 반환"""
#     return EventSourceResponse(progress_generator(space_id, service))

def get_progress_stream(space_id: str, service: Any = None):
    """진행률 스트림 반환"""
    return EventSourceResponse(
        progress_generator(space_id, service),
        headers={
            "X-Accel-Buffering": "no", 
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
            }
        )

# Result Service

# 진행률 저장소
result_progress_store: Dict[str, Dict[str, Any]] = {}

def update_result_progress(result_id: str, progress: int, data: Dict[str, Any] = None):
    """Result 전용 진행률 업데이트"""
    if result_id not in result_progress_store:
        result_progress_store[result_id] = {
            "progress": 0,
            "data": {}
        }
    
    result_progress_store[result_id]["progress"] = progress

    if data:
        result_progress_store[result_id]["data"] = data

async def result_progress_generator(result_id: str, service: Any = None):
    """Result Service SSE 진행률 생성기"""
    try:
        # 1. 초기 상태 확인
        if result_id not in result_progress_store:
            update_result_progress(result_id, 0)
        
        # 2. 처리 시작
        if service:
            # 비동기 처리 시작
            asyncio.create_task(service.process_grading())
        
        # 3. SSE 연결 시작
        while True:
            if result_id in result_progress_store:
                data = result_progress_store[result_id]
                progress = data["progress"]
                result_data = data.get("data", {})
                
                # 4. 진행 상태에 따른 이벤트 전송
                if progress == 100:
                    # Complete 응답 (API 명세 기준)
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "success": True,
                            "response": result_data,
                            "error": None
                        })
                    }
                    break
                    
                elif progress == -1:
                    # Error 응답
                    error_message = result_data.get("error", StatusMessage.ERROR)
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "success": False,
                            "response": None,
                            "error": error_message
                        })
                    }
                    break
                    
                else:
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "success": True,
                            "response": {
                                "progress": progress
                            },
                            "error": None
                        })
                    }
            
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"[result_progress_generator] SSE Error: {str(e)}")
        yield {
            "event": "error",
            "data": json.dumps({
                "success": False,
                "response": None,
                "error": f"SSE Error: {str(e)}"
            })
        }

def get_result_progress_stream(result_id: str, service: Any = None):
    """Result용 진행률 스트림 반환"""
    return EventSourceResponse(result_progress_generator(result_id, service))