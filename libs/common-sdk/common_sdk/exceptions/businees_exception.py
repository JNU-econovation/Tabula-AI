from fastapi import HTTPException, status


"""
인증
1. 잘못된 인증 토큰
2. 만료된 인증 토큰
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