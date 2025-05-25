import base64
from fastapi import Depends, Header
from jose import jwt, ExpiredSignatureError
from common_sdk.config import settings

from common_sdk.get_logger import get_logger

# 로그 설정
logger = get_logger()


# 인증을 위한 환경변수 세팅
ALGORITHM = "HS256"

# BASE64로 인코딩된 JWT_SECRET 디코딩
JWT_SECRET = base64.urlsafe_b64decode(settings.JWT_SECRET)


# Bearer token 추출 및 디코딩
def get_token_from_header(authorization: str = Header(...)):
    if not authorization:
        logger.error("Invalid token format")
        # 잘못된 인증 토큰 예외처리 
        raise Exception("Invalid token format")
    token = authorization.split("Bearer ")[1]
    return token


# Token에서 member id 가져오기
async def get_current_member(token: str = Depends(get_token_from_header)):
    
    if isinstance(token, dict):
        return token

    try:
        decoded_payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id: int = decoded_payload.get("sub")

        if user_id is None:
            logger.error("Token does not contain user_id")
            raise Exception("Token does not contain user_id")
        
        logger.debug(f"Decoded token user_id: {user_id}")
        return user_id

    except ExpiredSignatureError:
        # 만료된 인증 토큰 예외처리
        logger.error("Token expired")
        raise Exception("Token expired")