from fastapi import APIRouter
from common_sdk import get_logger

logger = get_logger()

router = APIRouter(
    prefix="/v1/spaces",
)

@router.post("/{space_id}/result")
async def upload_result():
    pass

