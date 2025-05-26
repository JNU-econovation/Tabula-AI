from fastapi import HTTPException, status


"""
- Authentication

1. 잘못된 인증 토큰
2. 만료된 인증 토큰
3. 빈 인증 토큰
"""
# 잘못된 인증 토큰
class InvalidJWT(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "SECURITY_401_1", 
                "reason": "잘못된 인증 토큰 형식입니다.", 
                "http_status": status.HTTP_401_UNAUTHORIZED
            }
        )

# 만료된 인증 토큰 
class ExpiredJWT(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "SECURITY_401_2", 
                "reason": "인증 토큰이 만료되었습니다.", 
                "http_status": status.HTTP_401_UNAUTHORIZED
            }
        )

# 빈 인증 토큰
class EmptyJWT(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "SECURITY_401_3", 
                "reason": "인증 토큰이 존재하지 않습니다.", 
                "http_status": status.HTTP_401_UNAUTHORIZED
            }
        )


"""
- Space

1. 필드 데이터 누락
"""
# 필드 데이터(langType, fileDomain) 누락
class MissingFieldData(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "SPACE_400_1", 
                "reason": "필드 데이터가 누락되었습니다.", 
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )


"""
- File

1. 학습 자료 파일 데이터(PDF) 누락
2. 학습 자료 파일 형식 미지원
3. 학습 자료 파일 용량 초과
4. 학습 결과물 파일 데이터(PDF / Image) 누락
5. 학습 결과물 파일 형식(PDF / Image) 미지원
6. 학습 결과물 파일 용량 초과
7. 학습 결과물 업로드 가능 페이지 초과
"""
# 파일 데이터(PDF) 누락
class MissingNoteFileData(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "FILE_400_1", 
                "reason": "파일 데이터(PDF) 누락입니다.", 
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )

# 파일 형식 미지원
class UnsupportedNoteFileFormat(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "FILE_400_2", 
                "reason": "파일 형식이 유효하지 않습니다.(지원되는 파일 형식: PDF).",
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )

# 파일 용량 초과
class NoteFileSizeExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_413_1", 
                "reason": "파일(PDF) 크기 허용 범위 초과입니다.(허용 범위: 5MB)",
                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            }
        )

# 학습 결과물 파일 데이터(PDF / Image) 누락
class MissingResultFileData(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "FILE_400_3", 
                "reason": "파일 데이터(PDF/Image) 누락입니다.", 
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )

# 학습 결과물 파일 형식(PDF / Image) 미지원
class UnsupportedResultFileFormat(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "FILE_400_4",
                "reason": "파일 형식이 유효하지 않습니다.(지원되는 파일 형식: PDF / Image)",
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )

# 학습 결과물 파일 용량 초과
class ResultFileSizeExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_413_2", 
                "reason": "파일(PDF / Image) 크기가 허용 범위 초과입니다.(허용 범위: 5MB)",
                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            }
        )

# 학습 결과물 업로드 가능 페이지 초과
class ResultFileUploadPageExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_413_3",
                "reason": "파일(PDF / Image)입력 페이지가 허용 범위 초과입니다.(허용 범위: 6페이지)",
                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            }
        )

"""
- Task

1. 요청 데이터 누락
2. 리소스 미존재
"""
# 요청 데이터(taskId) 누락
class MissingTaskId(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TASK_400_1", 
                "reason": "Request-Header에 taskId가 누락되었습니다.", 
                "http_status": status.HTTP_400_BAD_REQUEST
            }
        )

# 리소스 미존재
class TaskIdNotFound(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TASK_404_1", 
                "reason": "요청 데이터(taskId)에 해당하는 리소스가 존재하지 않습니다.", 
                "http_status": status.HTTP_404_NOT_FOUND
            }
        )