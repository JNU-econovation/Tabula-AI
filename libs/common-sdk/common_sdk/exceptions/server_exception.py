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

# 파일 찾기 실패(프롬프트 / 마크다운)
class FileNotFoundError(HTTPException):
    def __init__(self, file_path):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_3",
                "reason": f"File Not Found: {file_path}",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )

# 이미지 처리 실패
class ImageProcessingError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_4",
                "reason": "Image Processing Failed",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )

# 키워드 처리 실패
class KeywordProcessingError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_5",
                "reason": "Keyword Processing Failed",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )

# API KEY 오류
class APIKeyError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SERVER_500_6",
                "reason": "API Key Error",
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )