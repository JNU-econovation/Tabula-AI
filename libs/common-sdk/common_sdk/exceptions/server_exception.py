from fastapi import HTTPException, status

# 외부 연결 실패(MongoDB / S3)
class ExternalConnectionError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_1",
                "reason": "External Connection Failed(MongoDB / S3)",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )

# 적재 실패
class UploadFailedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_2",
                "reason": "Upload Failed(S3 / MongoDB)",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )