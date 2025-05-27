from pydantic import BaseModel
from typing import List, Optional

# 결과 이미지 모델
class ResultImage(BaseModel):
    id: int  # 이미지 순서
    resultImageUrl: str

# 응답 데이터 모델  
class ResponseData(BaseModel):
    resultId: int  # DB 기준 (result)
    results: List[ResultImage]

# 결과 응답 모델
class ResultResponse(BaseModel):
    success: bool
    response: Optional[ResponseData] = None
    error: Optional[str] = None