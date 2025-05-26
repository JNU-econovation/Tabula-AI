import uuid

from typing import Dict, Any
from fastapi import APIRouter, UploadFile, Form, Depends, File

from common_sdk import get_logger

# 로그 설정
logger = get_logger()

router = APIRouter(
    tags=["Result Service"]
)

@router.post("/{space_id}/result")
async def upload_result():
    pass

