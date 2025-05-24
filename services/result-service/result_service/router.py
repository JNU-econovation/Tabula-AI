from fastapi import APIRouter

router = APIRouter(
    prefix="/v1/spaces",
)

@router.post("/{space_id}/result")
async def upload_result():
    pass

