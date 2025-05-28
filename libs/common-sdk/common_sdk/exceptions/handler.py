from fastapi import Request, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from common_sdk.exceptions import (
    InvalidJWT, ExpiredJWT, EmptyJWT,
    MissingFieldData,
    MissingNoteFileData, UnsupportedNoteFileFormat, NoteFileSizeExceeded,
    MissingResultFileData, UnsupportedResultFileFormat, ResultFileSizeExceeded, ResultFileUploadPageExceeded,
    MissingSpaceId, SpaceIdNotFound
)
from common_sdk.exceptions import (
    ExternalConnectionError, UploadFailedError, FileNotFoundError, ImageProcessingError,
    KeywordProcessingError, APIKeyError
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
    app.add_exception_handler(ExternalConnectionError, server_exception_handler)
    app.add_exception_handler(UploadFailedError, server_exception_handler)
    app.add_exception_handler(FileNotFoundError, server_exception_handler)
    app.add_exception_handler(ImageProcessingError, server_exception_handler)
    app.add_exception_handler(KeywordProcessingError, server_exception_handler)
    app.add_exception_handler(APIKeyError, server_exception_handler)
    

    # 비즈니스 예외 핸들러 등록
    app.add_exception_handler(InvalidJWT, business_exception_handler)
    app.add_exception_handler(ExpiredJWT, business_exception_handler)
    app.add_exception_handler(EmptyJWT, business_exception_handler)
    app.add_exception_handler(MissingFieldData, business_exception_handler)
    app.add_exception_handler(MissingNoteFileData, business_exception_handler)
    app.add_exception_handler(UnsupportedNoteFileFormat, business_exception_handler)
    app.add_exception_handler(NoteFileSizeExceeded, business_exception_handler)
    app.add_exception_handler(MissingResultFileData, business_exception_handler)
    app.add_exception_handler(UnsupportedResultFileFormat, business_exception_handler)
    app.add_exception_handler(ResultFileSizeExceeded, business_exception_handler)
    app.add_exception_handler(ResultFileUploadPageExceeded, business_exception_handler)
    app.add_exception_handler(MissingSpaceId, business_exception_handler)
    app.add_exception_handler(SpaceIdNotFound, business_exception_handler)