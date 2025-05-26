import secrets
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from common_sdk.config import settings

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Swagger UI 접근 인증"""
    correct_username = secrets.compare_digest(credentials.username, settings.SWAGGER_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, settings.SWAGGER_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

router = APIRouter(
    tags=["Swagger"]
)

@router.get("/docs", response_class=HTMLResponse)
async def get_docs(username: str = Depends(get_current_username)) -> HTMLResponse:
    """Swagger UI 접근"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Tabula API Documentation"
    )

@router.get("/redoc", response_class=HTMLResponse)
async def get_redoc(username: str = Depends(get_current_username)) -> HTMLResponse:
    """ReDoc UI 접근"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Tabula API Documentation"
    )