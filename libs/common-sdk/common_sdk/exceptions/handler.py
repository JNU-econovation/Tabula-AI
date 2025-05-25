from fastapi import Request, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from errors.business_exception import (
    InvalidJWT, ExpiredJWT
)
from errors.server_exception import (
    FileAccessError
)

# 서버 예외 핸들러
async def server_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "response": None,
            "error": exc.detail
        }
    )

# 비즈니스 예외 핸들러
async def business_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "response": None,
            "error": exc.detail
        }
    )

def register_exception_handlers(app: FastAPI):
    # 서버 예외 핸들러 등록
    app.add_exception_handler(FileAccessError, server_exception_handler)

    # 비즈니스 예외 핸들러 등록
    app.add_exception_handler(InvalidJWT, business_exception_handler)
    app.add_exception_handler(ExpiredJWT, business_exception_handler)