from fastapi import HTTPException, status

# 파일 접근 오류
class FileAccessError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_1",
                "reason": "파일 접근 중 오류가 발생했습니다.",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )