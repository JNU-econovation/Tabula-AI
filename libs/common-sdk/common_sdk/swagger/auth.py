import os
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from apis.swagger import get_current_username


# # 환경에 따른 root_path 설정
# env = os.getenv("APP_ENV")
# if env == "prod":
#     root_path = "/prod"
# elif env == "dev":
#     root_path = "/dev"
# else:
#     root_path = ""

# router = APIRouter(
#     tags=["Swagger"]
# )

# @router.get("/docs", response_class=HTMLResponse)
# async def get_docs(username: str = Depends(get_current_username)) -> HTMLResponse:
#     if env == "prod":
#         # 운영 환경에서 접근 차단
#         raise HTTPException(status_code=404, detail="Swagger UI is not available in production")
#     return get_swagger_ui_html(
#         openapi_url=f"{root_path}/ai/v1/api/openapi.json",
#         title="Swagger UI",
#     )

# @router.get("/redocs", response_class=HTMLResponse)
# async def get_redoc(username: str = Depends(get_current_username)) -> HTMLResponse:
#     if env == "prod":
#         # 운영 환경에서 접근 차단
#         raise HTTPException(status_code=404, detail="ReDoc is not available in production")
#     return get_redoc_html(
#         openapi_url=f"{root_path}/ai/v1/api/openapi.json",
#         title="ReDoc",
#     )